#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
import uuid
from argparse import ArgumentParser, Namespace
from random import randint, random
from typing import List, Optional, Tuple, Union

import torch
import torchvision
import torchvision.transforms as T
from tqdm import tqdm

from arguments import GroupParams, ModelParams, OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene import GaussianModel, Scene
from utils.general_utils import safe_state
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def training(
    dataset: GroupParams,
    opt: GroupParams,
    pipe: GroupParams,
    testing_iterations: List[int],
    saving_iterations: List[int],
    checkpoint_path: Optional[str] = None,
    filter3d: bool = False,
    dense: bool = False,
    debug_from: int = -1,
) -> None:
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    model_path = dataset.model_path
    gaussians = GaussianModel(sh_degree=dataset.sh_degree)
    scales = [1.0, 2.0, 4.0, 8.0]
    scene = Scene(args=dataset, gaussians=gaussians, resolution_scales=scales)
    gaussians.training_setup(opt)

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path)
        model_params = checkpoint["gaussians"]
        first_iter = checkpoint["iteration"]
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing=True)
    iter_end = torch.cuda.Event(enable_timing=True)

    trainCameras = []
    testCameras = []
    for scale in scales:
        trainCameras += scene.getTrainCameras(scale).copy()
        testCameras += scene.getTestCameras(scale).copy()

    # highresolution index
    highresolution_index = list(range(len(scene.getTrainCameras(1.0))))

    gaussians.compute_3D_filter(cameras=trainCameras)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = trainCameras.copy()
        idx = randint(0, len(viewpoint_stack) - 1)
        viewpoint_cam = viewpoint_stack.pop(idx)

        # NOTE: a trick from mip-splatting
        # Pick a random high resolution camera
        if random() < 0.3 and dataset.sample_more_highres and len(highresolution_index) > 0:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index) - 1)]]

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, filter3d=filter3d)
        image, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"],
            render_pkg["viewspace_points"],
            render_pkg["visibility_filter"],
            render_pkg["radii"],
        )

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "#Splat": gaussians.get_xyz.shape[0]})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer=tb_writer,
                iteration=iteration,
                Ll1=Ll1,
                loss=loss,
                elapsed=iter_start.elapsed_time(iter_end),
                testing_iterations=testing_iterations,
                saving_iterations=saving_iterations,
                scene=scene,
                renderArgs=(pipe, background),
                filter3d=filter3d,
                model_path=model_path,
                scales=scales,
            )

            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = opt.max_screen_size if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, dense=dense)
                    gaussians.compute_3D_filter(cameras=trainCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)

            if iteration in saving_iterations:
                print(f"\n[ITER {iteration}] Saving Checkpoint")
                scene.save(iteration)


def prepare_output_and_logger(args: GroupParams) -> Optional[SummaryWriter]:
    if not args.model_path:
        if os.getenv("OAR_JOB_ID"):
            unique_str = os.getenv("OAR_JOB_ID")
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print(f"Output folder: {args.model_path}")
    os.makedirs(args.model_path, exist_ok=True)
    with open(os.path.join(args.model_path, "cfg_args"), "w") as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(
    tb_writer: Optional[SummaryWriter],
    iteration: int,
    Ll1: Union[float, torch.Tensor],
    loss: Union[float, torch.Tensor],
    elapsed: float,
    testing_iterations: List[int],
    saving_iterations: List[int],
    scene: Scene,
    renderArgs: Tuple[GroupParams, torch.Tensor],
    filter3d: bool = False,
    model_path: str = "",
    scales: List[float] = [1.0, 2.0, 4.0, 8.0],
) -> None:
    if tb_writer:
        tb_writer.add_scalar("train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar("train_loss_patches/total_loss", loss.item(), iteration)
        tb_writer.add_scalar("iter_time", elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = (
            {"name": "test", "cameras": {}},
            {"name": "train", "cameras": {}},
        )
        for scale in scales:
            validation_configs[0]["cameras"][scale] = scene.getTestCameras(scale).copy()
            validation_configs[1]["cameras"][scale] = []
            train_cameras = scene.getTrainCameras(scale).copy()
            if len(train_cameras) > 0:
                validation_configs[1]["cameras"][scale] = [train_cameras[idx % len(train_cameras)] for idx in range(5, 30, 5)]

        for config in validation_configs:
            name = config["name"]
            if config["cameras"] and len(config["cameras"]) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                l1_scale_test = {scale:0 for scale in scales}
                psnr_scale_test = {scale:0 for scale in scales}
                save_dir = os.path.join(model_path, "%06d" % iteration, config["name"])
                gt_dir = os.path.join(model_path, config["name"] + "_gt")
                os.makedirs(save_dir, exist_ok=True)
                os.makedirs(gt_dir, exist_ok=True)
                for scale in scales:
                    for idx, viewpoint in tqdm(enumerate(config["cameras"][scale]), desc=f"{config['name']}_{scale}", total=len(config["cameras"][scale]), leave=False):
                        render_dict = render(viewpoint, scene.gaussians, *renderArgs, filter3d=filter3d)
                        image = torch.clamp(render_dict["render"], 0.0, 1.0)
                        gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                        if tb_writer and (idx < 5):
                            tb_writer.add_images(f"{name}_view_{viewpoint.image_name}/render", image[None], global_step=iteration)
                            if iteration == testing_iterations[0]:
                                tb_writer.add_images(f"{name}_view_{viewpoint.image_name}/ground_truth", gt_image[None], global_step=iteration)
                        if iteration in saving_iterations:
                            image_name = viewpoint.image_name
                            if image_name.split(".")[-1].lower() not in ["png", "jpg", "jpeg"]:
                                image_name += ".png"
                            torchvision.utils.save_image(image, os.path.join(save_dir, image_name))
                            if iteration == testing_iterations[0]:
                                torchvision.utils.save_image(gt_image, os.path.join(gt_dir, image_name))
                        l1_val = l1_loss(image, gt_image).mean().double()
                        psnr_val = psnr(image, gt_image).mean().double()
                        l1_test += l1_val
                        psnr_test += psnr_val
                        l1_scale_test[scale] += l1_val
                        psnr_scale_test[scale] += psnr_val
                l1_test = l1_test / sum([len(config["cameras"][s]) for s in scales])
                psnr_test = psnr_test / sum([len(config["cameras"][s]) for s in scales])
                for scale in scales:
                    l1_scale_test[scale] = l1_scale_test[scale] / max(1, len(config["cameras"][scale]))
                    psnr_scale_test[scale] = psnr_scale_test[scale] / max(1, len(config["cameras"][scale]))
                text = f"\n[ITER {iteration}] Evaluating {name}: L1 {l1_test:.6f} PSNR {psnr_test:.6f}"
                for scale in scales:
                    text += f" PSNR_{scale} {psnr_scale_test[scale]:.6f}"
                print(text)
                if tb_writer:
                    tb_writer.add_scalar(f"{name}/loss_viewpoint - l1_loss", l1_test, iteration)
                    tb_writer.add_scalar(f"{name}/loss_viewpoint - psnr", psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar("total_points", scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument("--debug_from", type=int, default=-1)
    parser.add_argument("--detect_anomaly", action="store_true", default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--filter3d", action="store_true")
    parser.add_argument("--dense", action="store_true")
    parser.add_argument("--start_checkpoint", type=str, default=None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(
        dataset=lp.extract(args),
        opt=op.extract(args),
        pipe=pp.extract(args),
        testing_iterations=args.test_iterations,
        saving_iterations=args.save_iterations,
        checkpoint_path=args.start_checkpoint,
        filter3d=args.filter3d,
        dense=args.dense,
        debug_from=args.debug_from,
    )

    # All done
    print("\nTraining complete.")
