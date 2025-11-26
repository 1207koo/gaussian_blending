/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
    void render(
        const dim3 grid, const dim3 block,
        const int W, const int H,
        const float* bg_color,
        const float* colors,
        const uint32_t* point_list,
        const float3* means2D,
        const float4* cov_opacity,
        const float2* lambdas,
        const float4* nv1_nv2,
        const uint2* ranges,
        const uint32_t* n_contrib,
        const float* final_color,
        const float* dL_dpixels,
        float3* dL_dmean2D,
        float4* dL_dcov2D,
        float* dL_dopacity,
        float* dL_dcolors
    );

    void preprocess(
        const int P, const int D, const int M,
        const float focal_x, const float focal_y,
        const float tan_fovx, const float tan_fovy,
        const float3* means3D,
        const int* radii,
        const float* shs,
        const bool* clamped,
        const glm::vec3* scales,
        const glm::vec4* rotations,
        const float scale_modifier,
        const float* cov3Ds,
        const float* viewmatrix,
        const float* projmatrix,
        const glm::vec3* campos,
        const float3* dL_dmean2D,
        const float* dL_dcov,
        glm::vec3* dL_dmean3D,
        float* dL_dcolor,
        float* dL_dcov3D,
        float* dL_dsh,
        glm::vec3* dL_dscale,
        glm::vec4* dL_drot
    );
}

#endif
