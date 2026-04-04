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

#pragma once

#include <iostream>
#include <vector>
#include "rasterizer.h"
#include <cuda_runtime_api.h>

namespace CudaRasterizer
{
	template <typename T>
	static void obtain(char*& chunk, T*& ptr, std::size_t count, std::size_t alignment)
	{
		std::size_t offset = (reinterpret_cast<std::uintptr_t>(chunk) + alignment - 1) & ~(alignment - 1);
		ptr = reinterpret_cast<T*>(offset);
		chunk = reinterpret_cast<char*>(ptr + count);
	}

	struct GeometryState
	{
		float* depths;
		bool* clamped;
		int* internal_radii;
		float3* means2D;
		float* cov3D;
		float4* cov_opacity;
		float2* lambdas;
		float4* nv1_nv2;
		float* rgb;

		static GeometryState fromChunk(char*& chunk, size_t P);
	};

	struct ImageState
	{
		uint32_t* n_contrib;
		float* final_color;
		uint2* ranges;
		// Per-tile binning
		uint32_t* tile_counts;
		uint32_t* tile_offsets;       // inclusive prefix sum
		uint32_t* tile_scatter_cnt;   // atomic counters for scatter
		size_t tile_scan_size;
		char* tile_scanning_space;

		static ImageState fromChunk(char*& chunk, size_t N, size_t num_tiles);
	};

	struct BinningState
	{
		size_t sorting_size;
		uint32_t* point_list;          // final sorted gaussian IDs
		uint32_t* point_list_unsorted; // scatter target (gaussian IDs)
		float* depth_keys_unsorted;    // scatter target (depths)
		float* depth_keys_sorted;      // sorted depths (temp)
		char* list_sorting_space;      // CUB segmented sort scratch

		static BinningState fromChunk(char*& chunk, size_t P, size_t num_tiles);
	};

	template<typename T>
	size_t required(size_t P)
	{
		char* size = nullptr;
		T::fromChunk(size, P);
		return ((size_t)size) + 128;
	}

	template<typename T>
	size_t required(size_t P, size_t num_tiles)
	{
		char* size = nullptr;
		T::fromChunk(size, P, num_tiles);
		return ((size_t)size) + 128;
	}
};
