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

#include "rasterizer_impl.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cub/cub.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <cub/device/device_segmented_radix_sort.cuh>
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#include "auxiliary.h"
#include "forward.h"
#include "backward.h"

// Wrapper method to call auxiliary coarse frustum containment test.
__global__ void checkFrustum(
    const int P,
    const float* orig_points,
    const float* viewmatrix,
    const float* projmatrix,
    bool* present
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    float3 p_view;
    present[idx] = in_frustum(idx, orig_points, viewmatrix, projmatrix, false, p_view);
}

// Phase 1: Count Gaussians per tile using tight AABB and atomics.
__global__ void countPerTile(
    const dim3 grid,
    const int P,
    const float3* __restrict__ points_xy,
    const int* __restrict__ radii,
    const float4* __restrict__ cov_opacity,
    uint32_t* __restrict__ tile_counts
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    if (radii[idx] > 0) {
        uint2 rect_min, rect_max;
        const float4 co = cov_opacity[idx];
        getRectTight(points_xy[idx], co.x, co.z, rect_min, rect_max, grid);

        for (uint32_t y = rect_min.y; y < rect_max.y; y++) {
            for (uint32_t x = rect_min.x; x < rect_max.x; x++) {
                atomicAdd(&tile_counts[y * grid.x + x], 1);
            }
        }
    }
}

// Phase 3: Scatter Gaussians into per-tile buckets.
// Uses tight AABB (must match countPerTile).
__global__ void scatterToTiles(
    const dim3 grid,
    const int P,
    const float3* __restrict__ points_xy,
    const float* __restrict__ depths,
    const int* __restrict__ radii,
    const float4* __restrict__ cov_opacity,
    const uint32_t* __restrict__ tile_offsets,   // inclusive prefix sum
    uint32_t* __restrict__ tile_scatter_cnt,
    float* __restrict__ depth_keys,
    uint32_t* __restrict__ gaussian_ids
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= P)
        return;

    if (radii[idx] > 0) {
        uint2 rect_min, rect_max;
        const float4 co = cov_opacity[idx];
        getRectTight(points_xy[idx], co.x, co.z, rect_min, rect_max, grid);

        for (uint32_t y = rect_min.y; y < rect_max.y; y++) {
            for (uint32_t x = rect_min.x; x < rect_max.x; x++) {
                const uint32_t tile_id = y * grid.x + x;
                const uint32_t pos = atomicAdd(&tile_scatter_cnt[tile_id], 1);
                const uint32_t base = (tile_id == 0) ? 0 : tile_offsets[tile_id - 1];
                depth_keys[base + pos] = depths[idx];
                gaussian_ids[base + pos] = idx;
            }
        }
    }
}

// Fill ranges array from inclusive prefix sum offsets.
__global__ void fillRanges(
    const int num_tiles,
    const uint32_t* __restrict__ tile_offsets,
    uint2* __restrict__ ranges
) {
    auto idx = cg::this_grid().thread_rank();
    if (idx >= num_tiles)
        return;

    uint32_t start = (idx == 0) ? 0 : tile_offsets[idx - 1];
    uint32_t end = tile_offsets[idx];
    ranges[idx] = { start, end };
}

// Mark Gaussians as visible/invisible
void CudaRasterizer::Rasterizer::markVisible(
    const int P,
    float* means3D,
    float* viewmatrix,
    float* projmatrix,
    bool* present
) {
    checkFrustum<<<(P + 255) / 256, 256>>>(P, means3D, viewmatrix, projmatrix, present);
}

CudaRasterizer::GeometryState CudaRasterizer::GeometryState::fromChunk(char*& chunk, const size_t P) {
    GeometryState geom;
    obtain(chunk, geom.depths, P, 128);
    obtain(chunk, geom.clamped, P * 3, 128);
    obtain(chunk, geom.internal_radii, P, 128);
    obtain(chunk, geom.means2D, P, 128);
    obtain(chunk, geom.cov3D, P * 6, 128);
    obtain(chunk, geom.cov_opacity, P, 128);
    obtain(chunk, geom.lambdas, P, 128);
    obtain(chunk, geom.nv1_nv2, P, 128);
    obtain(chunk, geom.rgb, P * 3, 128);
    return geom;
}

CudaRasterizer::ImageState CudaRasterizer::ImageState::fromChunk(char*& chunk, const size_t N, const size_t num_tiles) {
    ImageState img;
    obtain(chunk, img.n_contrib, N, 128);
    obtain(chunk, img.final_color, N * 3, 128);
    obtain(chunk, img.ranges, num_tiles, 128);
    obtain(chunk, img.tile_counts, num_tiles, 128);
    obtain(chunk, img.tile_offsets, num_tiles, 128);
    obtain(chunk, img.tile_scatter_cnt, num_tiles, 128);
    cub::DeviceScan::InclusiveSum(nullptr, img.tile_scan_size, img.tile_counts, img.tile_offsets, num_tiles);
    obtain(chunk, img.tile_scanning_space, img.tile_scan_size, 128);
    return img;
}

CudaRasterizer::BinningState CudaRasterizer::BinningState::fromChunk(char*& chunk, const size_t P, const size_t num_tiles) {
    BinningState binning;
    obtain(chunk, binning.point_list, P, 128);
    obtain(chunk, binning.point_list_unsorted, P, 128);
    obtain(chunk, binning.depth_keys_unsorted, P, 128);
    obtain(chunk, binning.depth_keys_sorted, P, 128);
    cub::DeviceSegmentedRadixSort::SortPairs(
        nullptr, binning.sorting_size,
        binning.depth_keys_unsorted, binning.depth_keys_sorted,
        binning.point_list_unsorted, binning.point_list,
        P, num_tiles, (uint32_t*)nullptr, (uint32_t*)nullptr);
    obtain(chunk, binning.list_sorting_space, binning.sorting_size, 128);
    return binning;
}

// Forward rendering procedure
int CudaRasterizer::Rasterizer::forward(
    const int P, int D, int M,
    const int width, int height,
    const float tan_fovx, float tan_fovy,
    const float scale_modifier,
    std::function<char* (size_t)> geometryBuffer,
    std::function<char* (size_t)> binningBuffer,
    std::function<char* (size_t)> imageBuffer,
    const float* background,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* opacities,
    const float* scales,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* cam_pos,
    const bool prefiltered,
    float* out_color,
    int* radii,
    bool debug
) {
    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    size_t chunk_size = required<GeometryState>(P);
    char* chunkptr = geometryBuffer(chunk_size);
    GeometryState geomState = GeometryState::fromChunk(chunkptr, P);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    dim3 block(BLOCK_X, BLOCK_Y, 1);
    const int num_tiles = tile_grid.x * tile_grid.y;

    size_t img_chunk_size = required<ImageState>(width * height, num_tiles);
    char* img_chunkptr = imageBuffer(img_chunk_size);
    ImageState imgState = ImageState::fromChunk(img_chunkptr, width * height, num_tiles);

    if (NUM_CHANNELS != 3 && colors_precomp == nullptr)
        throw std::runtime_error("For non-RGB, provide precomputed Gaussian colors!");

    // Preprocess
    CHECK_CUDA(FORWARD::preprocess(
        tile_grid, P, D, M, width, height,
        focal_x, focal_y, tan_fovx, tan_fovy,
        means3D, (glm::vec3*)scales, scale_modifier,
        (glm::vec4*)rotations, opacities, shs,
        cov3D_precomp, colors_precomp,
        viewmatrix, projmatrix, (glm::vec3*)cam_pos,
        radii, geomState.clamped, geomState.means2D,
        geomState.depths, geomState.cov3D, geomState.rgb,
        geomState.cov_opacity, geomState.lambdas, geomState.nv1_nv2,
        prefiltered
    ), debug)

    // === Per-tile binning pipeline ===

    // Phase 1: Count Gaussians per tile
    CHECK_CUDA(cudaMemset(imgState.tile_counts, 0, num_tiles * sizeof(uint32_t)), debug);
    countPerTile<<<(P + 255) / 256, 256>>>(
        tile_grid, P, geomState.means2D, radii, geomState.cov_opacity, imgState.tile_counts);
    CHECK_CUDA(, debug)

    // Phase 2: Inclusive prefix sum on tile counts
    CHECK_CUDA(cub::DeviceScan::InclusiveSum(
        imgState.tile_scanning_space, imgState.tile_scan_size,
        imgState.tile_counts, imgState.tile_offsets, num_tiles), debug)

    // Get total instance count
    int num_rendered;
    CHECK_CUDA(cudaMemcpy(&num_rendered, imgState.tile_offsets + num_tiles - 1,
        sizeof(int), cudaMemcpyDeviceToHost), debug);

    // Allocate binning buffers
    size_t binning_chunk_size = required<BinningState>(num_rendered, num_tiles);
    char* binning_chunkptr = binningBuffer(binning_chunk_size);
    BinningState binningState = BinningState::fromChunk(binning_chunkptr, num_rendered, num_tiles);

    // Phase 3: Scatter Gaussians to per-tile buckets
    CHECK_CUDA(cudaMemset(imgState.tile_scatter_cnt, 0, num_tiles * sizeof(uint32_t)), debug);
    scatterToTiles<<<(P + 255) / 256, 256>>>(
        tile_grid, P, geomState.means2D, geomState.depths, radii,
        geomState.cov_opacity, imgState.tile_offsets, imgState.tile_scatter_cnt,
        binningState.depth_keys_unsorted, binningState.point_list_unsorted);
    CHECK_CUDA(, debug)

    // Phase 4: Sort per tile by depth using CUB segmented sort
    CHECK_CUDA(cudaMemset(imgState.ranges, 0, num_tiles * sizeof(uint2)), debug);
    if (num_rendered > 0) {
        fillRanges<<<(num_tiles + 255) / 256, 256>>>(
            num_tiles, imgState.tile_offsets, imgState.ranges);
        CHECK_CUDA(, debug)

        // Build begin_offsets: reuse tile_scatter_cnt (no longer needed)
        // begin[0]=0, begin[i]=tile_offsets[i-1]
        if (num_tiles > 1) {
            CHECK_CUDA(cudaMemcpy(
                imgState.tile_scatter_cnt + 1,
                imgState.tile_offsets,
                (num_tiles - 1) * sizeof(uint32_t),
                cudaMemcpyDeviceToDevice), debug);
        }
        CHECK_CUDA(cudaMemset(imgState.tile_scatter_cnt, 0, sizeof(uint32_t)), debug);

        CHECK_CUDA(cub::DeviceSegmentedRadixSort::SortPairs(
            binningState.list_sorting_space,
            binningState.sorting_size,
            binningState.depth_keys_unsorted, binningState.depth_keys_sorted,
            binningState.point_list_unsorted, binningState.point_list,
            num_rendered, num_tiles,
            imgState.tile_scatter_cnt,  // d_begin_offsets
            imgState.tile_offsets       // d_end_offsets
        ), debug)
    }

    // Render
    const float* feature_ptr = colors_precomp != nullptr ? colors_precomp : geomState.rgb;
    CHECK_CUDA(FORWARD::render(
        tile_grid, block, width, height, background, feature_ptr,
        binningState.point_list, geomState.means2D, geomState.cov_opacity,
        geomState.lambdas, geomState.nv1_nv2, imgState.ranges,
        imgState.n_contrib, imgState.final_color, out_color), debug)

    return num_rendered;
}

// Backward pass
void CudaRasterizer::Rasterizer::backward(
    const int P, int D, int M, int R,
    const int width, int height,
    const float tan_fovx, float tan_fovy,
    const float scale_modifier,
    const float* background,
    const float* means3D,
    const float* shs,
    const float* colors_precomp,
    const float* scales,
    const float* rotations,
    const float* cov3D_precomp,
    const float* viewmatrix,
    const float* projmatrix,
    const float* campos,
    const int* radii,
    char* geom_buffer,
    char* binning_buffer,
    char* img_buffer,
    const float* dL_dpix,
    float* dL_dmean2D,
    float* dL_dcov,
    float* dL_dopacity,
    float* dL_dcolor,
    float* dL_dmean3D,
    float* dL_dcov3D,
    float* dL_dsh,
    float* dL_dscale,
    float* dL_drot,
    bool debug
) {
    GeometryState geomState = GeometryState::fromChunk(geom_buffer, P);

    const dim3 tile_grid((width + BLOCK_X - 1) / BLOCK_X, (height + BLOCK_Y - 1) / BLOCK_Y, 1);
    const dim3 block(BLOCK_X, BLOCK_Y, 1);
    const int num_tiles = tile_grid.x * tile_grid.y;

    BinningState binningState = BinningState::fromChunk(binning_buffer, R, num_tiles);
    ImageState imgState = ImageState::fromChunk(img_buffer, width * height, num_tiles);

    if (radii == nullptr)
        radii = geomState.internal_radii;

    const float focal_y = height / (2.0f * tan_fovy);
    const float focal_x = width / (2.0f * tan_fovx);

    const float* color_ptr = (colors_precomp != nullptr) ? colors_precomp : geomState.rgb;
    CHECK_CUDA(BACKWARD::render(
        tile_grid, block, width, height, background, color_ptr,
        binningState.point_list, geomState.means2D, geomState.cov_opacity,
        geomState.lambdas, geomState.nv1_nv2, imgState.ranges,
        imgState.n_contrib, imgState.final_color,
        dL_dpix, (float3*)dL_dmean2D, (float4*)dL_dcov,
        dL_dopacity, dL_dcolor), debug)

    const float* cov3D_ptr = (cov3D_precomp != nullptr) ? cov3D_precomp : geomState.cov3D;
    CHECK_CUDA(BACKWARD::preprocess(
        P, D, M, focal_x, focal_y, tan_fovx, tan_fovy,
        (float3*)means3D, radii, shs, geomState.clamped,
        (glm::vec3*)scales, (glm::vec4*)rotations, scale_modifier,
        cov3D_ptr, viewmatrix, projmatrix, (glm::vec3*)campos,
        (float3*)dL_dmean2D, dL_dcov, (glm::vec3*)dL_dmean3D,
        dL_dcolor, dL_dcov3D, dL_dsh,
        (glm::vec3*)dL_dscale, (glm::vec4*)dL_drot), debug)
}
