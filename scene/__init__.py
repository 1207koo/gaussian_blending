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

import json
import os
import random
from typing import List, Optional

from arguments import ModelParams
from scene.cameras import Camera
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from utils.camera_utils import camera_to_JSON, cameraList_from_camInfos
from utils.system_utils import searchForMaxIteration


class Scene:
    gaussians: GaussianModel

    def __init__(
        self,
        args: ModelParams,
        gaussians: GaussianModel,
        load_iteration: Optional[int] = None,
        shuffle: bool = True,
        resolution_scales=[1.0, 2.0, 4.0, 8.0],
    ) -> None:
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(
                    os.path.join(self.model_path, "point_cloud")
                )
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_type = ""
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_type = "colmap"
            scene_info = sceneLoadTypeCallbacks["Colmap"](args.source_path, args.images, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
            scene_type = "blender"
            print("Found transforms_train.json file, assuming Blender data set!")
            scene_info = sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.eval)
        elif os.path.exists(os.path.join(args.source_path, "metadata.json")):
            scene_type = "ms-blender"
            print("Found metadata.json file, assuming multi scale Blender data set!")
            scene_info = sceneLoadTypeCallbacks["MS-Blender"](args.source_path, args.white_background, args.eval, args.train_res)
        else:
            assert False, "Could not recognize scene type!"

        if not self.loaded_iter:
            with open(scene_info.ply_path, "rb") as src_file, open(os.path.join(self.model_path, "input.ply"), "wb") as dest_file:
                dest_file.write(src_file.read())
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), "w") as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]


        if scene_type == "ms-blender":
            for resolution_scale in resolution_scales:
                print(f"Loading Training Cameras with resolution scale: {resolution_scale}")
                if resolution_scale in args.train_res:
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos([cam for cam in scene_info.train_cameras if cam.res == resolution_scale], 1.0, args)
                else:
                    print("Skipping loading of training cameras for resolution scale: ", resolution_scale)
                    self.train_cameras[resolution_scale] = []
                print(f"Loading Test Cameras with resolution scale: {resolution_scale}")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos([cam for cam in scene_info.test_cameras if cam.res == resolution_scale], 1.0, args)
        else:
            for resolution_scale in resolution_scales:
                print(f"Loading Training Cameras with resolution scale: {resolution_scale}")
                if resolution_scale in args.train_res:
                    self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
                else:
                    print("Skipping loading of training cameras for resolution scale: ", resolution_scale)
                    self.train_cameras[resolution_scale] = []
                print(f"Loading Test Cameras with resolution scale: {resolution_scale}")
                self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
                for cam in self.train_cameras[resolution_scale]:
                    img_name = cam.image_name
                    if img_name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
                        img_name = img_name.rsplit('.', 1)[0] + '_' + str(resolution_scale) + '.' + img_name.rsplit('.', 1)[-1]
                    else:
                        img_name += '_' + str(resolution_scale) + '.png'
                    cam.image_name = img_name
                for cam in self.test_cameras[resolution_scale]:
                    img_name = cam.image_name
                    if img_name.split('.')[-1].lower() in ['png', 'jpg', 'jpeg']:
                        img_name = img_name.rsplit('.', 1)[0] + '_' + str(resolution_scale) + '.' + img_name.rsplit('.', 1)[-1]
                    else:
                        img_name += '_' + str(resolution_scale) + '.png'
                    cam.image_name = img_name

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"))
        else:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)

    def save(self, iteration: int) -> None:
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale: float = -1.0) -> List[Camera]:
        if scale == -1.0:
            data = []
            for scale in self.train_cameras.keys():
                data.extend(self.train_cameras[scale])
            return data
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = -1.0) -> List[Camera]:
        if scale == -1.0:
            data = []
            for scale in self.test_cameras.keys():
                data.extend(self.test_cameras[scale])
            return data
        return self.test_cameras[scale]
