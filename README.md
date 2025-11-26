# <p align="center"> **Gaussian Blending**<br>Rethinking Alpha Blending in 3D Gaussian Splatting<br>[AAAI 2026] </p>

####  <p align="center"> [Junseo Koo](https://1207koo.github.io), [Jinseo Jeong](https://www.jinseo.kr), [Gunhee Kim](https://vision.snu.ac.kr/gunhee)</p>

#### <p align="center">[Paper (coming soon)](https://vision.snu.ac.kr/) | [arXiv](https://arxiv.org/abs/2511.15102) | [Project Page](https://1207koo.github.io/html/gaussianblending/)</p>


<p align="center">
  <img width="90%" src="assets/rendering_graph.png"/>
</p>


## Installation
Our code is tested with Python 3.8 with CUDA 12.4.
```bash
git clone https://github.com/1207koo/gaussian_blending.git
cd gaussian_blending

conda create -y -n gaussian_blending python=3.8
conda activate gaussian_blending

pip install -r requirements.txt
pip install submodules/simple-knn
pip install submodules/gaussian-blending
```


## Dataset
Our experiments are conducted on multi-scale Blender dataset and multi-scale Mip-NeRF 360 dataset.
You can download the datasets from their official websites: [Blender (nerf_synthetic)](https://www.matthewtancik.com/nerf), [Mip-NeRF 360](https://jonbarron.info/mipnerf360/)
Please download and extract the datasets into the `data/` directory as `data/nerf_synthetic` and `data/mipnerf360`, respectively.

### Multi-scale Blender dataset
To create a multi-scale Blender dataset, you can use the provided script `convert_blender_data.py` to convert the original single-scale Blender dataset into a multi-scale version.
```bash
# python convert_blender_data.py --blender_dir <BLENDER_DIR> --out_dir <BLENDER_MULTI_DIR>
python convert_blender_data.py --blender_dir data/nerf_synthetic --out_dir data/nerf_synthetic_multi
```


## Training
By default, our code performs multi-scale training and multi-scale testing.
To perform single-scale training and multi-scale testing, you can use the `--train_res` argument to specify the training resolution.
For example, `--train_res 1.0` indicates training at the original resolution only (zoom-out setting), while `--train_res 8.0` indicates training at 1/8 resolution only (zoom-in setting).

### Multi-scale Blender dataset
```bash
# python train.py -m <OUTPUT_DIR> -s <DATA_DIR> --white_background --eval --sample_more_highres --train_res <LIST_OF_TRAIN_RESOLUTIONS>

# Single-scale training and multi-scale testing (zoom-out setting)
python train.py -m output/blender_lego_stmt1 -s data/nerf_synthetic_multi/lego --white_background --eval --train_res 1.0
# Single-scale training and multi-scale testing (zoom-in setting)
python train.py -m output/blender_lego_stmt8 -s data/nerf_synthetic_multi/lego --white_background --eval --train_res 8.0
# Multi-scale training and multi-scale testing
python train.py -m output/blender_lego_mtmt -s data/nerf_synthetic_multi/lego --white_background --eval --sample_more_highres
```

### Mip-NeRF 360 dataset
```bash
# use '-r 4' for outdoor scenes, and '-r 2' for indoor scenes
# python train.py -m <OUTPUT_DIR> -s <DATA_DIR> --white_background --eval --sample_more_highres -r <DEFAULT_RESOLUTION> --train_res <LIST_OF_TRAIN_RESOLUTIONS>

# Single-scale training and multi-scale testing (zoom-out setting)
python train.py -m output/mipnerf360_bicycle_stmt1 -s data/mipnerf360/bicycle --white_background --eval -r 4 --train_res 1.0
# Single-scale training and multi-scale testing (zoom-in setting)
python train.py -m output/mipnerf360_bicycle_stmt8 -s data/mipnerf360/bicycle --white_background --eval -r 4 --train_res 8.0
# Multi-scale training and multi-scale testing
python train.py -m output/mipnerf360_bicycle_mtmt -s data/mipnerf360/bicycle --white_background --eval --sample_more_highres -r 4
```


## Rendering
You can render images using a trained model with the following command:
```bash
# python render.py -m <OUTPUT_DIR> -s <DATA_DIR> --white_background --eval --train_res <LIST_OF_TRAIN_RESOLUTIONS> -r <DEFAULT_RESOLUTION> --lpips --vis
python render.py -m output/blender_lego_mtmt -s data/nerf_synthetic_multi/lego --white_background --eval --lpips --vis
```

## Drop-in Replacement
You can easily integrate our Gaussian Blending as a drop-in replacement into existing 3DGS-based models.
1. Instead of using the original rendering module `submodules/diff-gaussian-rasterization`, replace it with our `submodules/gaussian-blending` module.
2. install the `submodules/gaussian-blending` module using the following command:
```bash
pip install submodules/gaussian-blending
```
3. Modify the rendering code `gaussian_renderer/__init__.py` to use `GaussianRasterizationSettings` and `GaussianRasterizer` from our module:
```python
from gaussian_blending import GaussianRasterizationSettings, GaussianRasterizer
```
4. And it's done! You can now use Gaussian Blending in your existing 3DGS-based model.


## Acknowledgment
Our code is built upon the following repositories:
- [3DGS](https://github.com/graphdeco-inria/gaussian-splatting)
- [Analytic-Splatting](https://github.com/lzhnb/Analytic-Splatting)


## Citation
If you find our work useful in your research, please consider citing our paper:
```txt
@inproceedings{koo2026gb,
    author = {Koo, Junseo and Jeong, Jinseo and Kim, Gunhee},
    title  = {{Gaussian Blending: Rethinking Alpha Blending in 3D Gaussian Splatting}},
    booktitle = {Proceedings of the AAAI conference on artificial intelligence},
    year = {2026},
}
```
