#include "cuda_functions.h"
#include <opencv2/cudev.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>

#define SPATIAL_BLOCK_SIZE 8

static __global__ void spatial_map_impl(const cv::cudev::PtrStepSz<f32> input, 
										cv::cudev::PtrStepSz<f32> output) {
	__shared__ f32 pixel_variation[(SPATIAL_BLOCK_SIZE - 1) * (SPATIAL_BLOCK_SIZE - 1)];
	const u32 x = SPATIAL_BLOCK_SIZE * blockIdx.x + threadIdx.x;
	const u32 y = SPATIAL_BLOCK_SIZE * blockIdx.y + threadIdx.y;

	if ((x < (input.cols - 1)) && (y < (input.rows - 1))) {
		f32 temp_sum = 0.f;
		temp_sum += std::abs(input.ptr(y    )[x    ] - input.ptr(y    )[x + 1]);
		temp_sum += std::abs(input.ptr(y    )[x    ] - input.ptr(y + 1)[x    ]);
		temp_sum += std::abs(input.ptr(y    )[x    ] - input.ptr(y + 1)[x + 1]);
		temp_sum += std::abs(input.ptr(y + 1)[x + 1] - input.ptr(y + 1)[x    ]);
		temp_sum += std::abs(input.ptr(y + 1)[x    ] - input.ptr(y    )[x + 1]);
		temp_sum += std::abs(input.ptr(y + 1)[x + 1] - input.ptr(y    )[x + 1]);
		pixel_variation[threadIdx.x + threadIdx.y * (SPATIAL_BLOCK_SIZE - 1)] = temp_sum;
	}
	
	__syncthreads();
	if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
		f32 max_variation = 0.f;
		for (u32 i = 0; i < ((SPATIAL_BLOCK_SIZE - 1) * (SPATIAL_BLOCK_SIZE - 1)); ++i) {
			if (max_variation < pixel_variation[i]) {
				max_variation = pixel_variation[i];
			}
		}
		// Each pixel ranges from 0 - 255, so divide by 255 to make it from 0 - 1
		// The maximum value of total variation for each 2x2 block is 4
		// Normalize max_variation to be from 0 to 1. 
		max_variation /= (255.f * 4.f);
		output.ptr(blockIdx.y)[blockIdx.x] = max_variation;
	}
}

void cuda_spatial_map(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream) {
	dim3 blocks(input.cols / SPATIAL_BLOCK_SIZE, input.rows / SPATIAL_BLOCK_SIZE);
	dim3 threads_per_block(SPATIAL_BLOCK_SIZE - 1, SPATIAL_BLOCK_SIZE - 1);
	
	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	spatial_map_impl<<<blocks,threads_per_block,0,cuda_stream>>>(input, output);
}

#define LUMINANCE_BLOCK_SIZE    32
#define LUMINANCE_DELTA_BLOCK   (LUMINANCE_BLOCK_SIZE  / 4)
#define LUMINANCE_OFFSET        (LUMINANCE_DELTA_BLOCK / 2)

static __global__ void luminance_map_impl(const cv::cudev::PtrStepSz<f32> input, 
										  cv::cudev::PtrStepSz<f32> output) {
	__shared__ f32 pixel_luminance[LUMINANCE_BLOCK_SIZE * LUMINANCE_BLOCK_SIZE];
	const u32 x = LUMINANCE_DELTA_BLOCK * blockIdx.x + threadIdx.x;
	const u32 y = LUMINANCE_DELTA_BLOCK * blockIdx.y + threadIdx.y;

	f32 pixel = input.ptr(y + LUMINANCE_OFFSET)[x + LUMINANCE_OFFSET];
	pixel *= 0.0364f;
	pixel += 0.7656f;
	pixel = pow(pixel, 2.2f);
	pixel_luminance[threadIdx.x + threadIdx.y * LUMINANCE_BLOCK_SIZE] = pixel;

	__syncthreads();
	if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
		f32 max_luminance = pixel_luminance[0];
		f32 min_luminance = pixel_luminance[0];
		f32 luminance_sum = 0.f;
		for (u32 i = 0; i < (LUMINANCE_BLOCK_SIZE * LUMINANCE_BLOCK_SIZE); ++i) {
			if (max_luminance < pixel_luminance[i]) {
				max_luminance = pixel_luminance[i];
			}
			if (min_luminance > pixel_luminance[i]) {
				min_luminance = pixel_luminance[i];
			}
			luminance_sum += pixel_luminance[i];
		}
		f32 mean_luminance = luminance_sum / (LUMINANCE_BLOCK_SIZE * LUMINANCE_BLOCK_SIZE);
		if (((max_luminance - min_luminance) > 5.f) && (mean_luminance > 2.f)) {
			output.ptr(blockIdx.y)[blockIdx.x] = 1.f;
		} else {
			output.ptr(blockIdx.y)[blockIdx.x] = 0.f;
		}
	}
}

void cuda_luminance_map(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream) {
	dim3 blocks(input.cols / LUMINANCE_DELTA_BLOCK, input.rows / LUMINANCE_DELTA_BLOCK);
	dim3 threads_per_block(LUMINANCE_BLOCK_SIZE, LUMINANCE_BLOCK_SIZE);
	
	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	luminance_map_impl<<<blocks,threads_per_block,0,cuda_stream>>>(input, output);
}

#define RADIAN_STEPS 36
#define RADIAN_SAMPLES (RADIAN_STEPS - 1)

static __global__ void polar_average_impl(const cv::cudev::PtrStepSz<f32> input_magnitudes, 
										  u32 input_n, 
										  cv::cudev::PtrStepSz<f32> output_magnitude_sums, 
										  cv::cudev::PtrStepSz<f32> output_frequencies) {
	const f32 two_pi = atanf(1.f) * 8.f;

	__shared__ f32 magnitude_at_angles[RADIAN_SAMPLES];

	const u32 radius   = blockIdx.x + 1;
	const u32 angle    = threadIdx.x * (two_pi / RADIAN_STEPS);

	f32 x = radius * sinf(angle); // x coordinate
	f32 y = radius * cosf(angle); // y coordinate
	
	f32 x1 = std::copysign(std::floor(std::abs(x)), x); // aliasing
	f32 x2 = std::copysign(std::ceil(std::abs(x)), x);
	f32 y1 = std::copysign(std::floor(std::abs(y)), y);
	f32 y2 = std::copysign(std::ceil(std::abs(y)), y);
	
	f32 ex = std::abs(x - x1);
	f32 ey = std::abs(y - y1);
	if (x2 < 0) {
		ex = std::abs(x - x2);
		if (x1 < 0) {
			x1 = input_n + x1;
		}
		x2 = input_n + x2;
	}
	
	if (y2 < 0) {
		ey = std::abs(y - y2);
		if (y1 < 0) {
			y1 = input_n + y1;
		}
		y2 = input_n + y2;
	}
	
	f32 f11 = input_magnitudes.ptr((u32)y1)[(u32)x1];
	f32 f12 = input_magnitudes.ptr((u32)y2)[(u32)x1];
	f32 f21 = input_magnitudes.ptr((u32)y1)[(u32)x2];
	f32 f22 = input_magnitudes.ptr((u32)y2)[(u32)x2];
	
	magnitude_at_angles[threadIdx.x] = (f21 - f11) * ex * (1.f - ey) + (f12 - f11) * (1.f - ex) * ey + (f22 - f11) * ex * ey + f11;

	__syncthreads();
	if (threadIdx.x == 0) {
		f32 magnitude_at_radius = 0.f;
		for (u32 i = 0; i < RADIAN_SAMPLES; ++i) {
			magnitude_at_radius += magnitude_at_angles[i];
		}
		magnitude_at_radius /= RADIAN_STEPS;
		output_magnitude_sums.ptr(blockIdx.x)[0] = magnitude_at_radius;
		output_frequencies.ptr(blockIdx.x)[0] = (0.5f / (input_n / 2)) * (blockIdx.x + 1);
	}
}

void cuda_polar_average(const cv::cuda::GpuMat& input_magnitudes, cv::cuda::GpuMat& output_magnitude_sums, cv::cuda::GpuMat& output_frequencies, cv::cuda::Stream& stream) {
	u32 n = std::min(input_magnitudes.rows, input_magnitudes.cols);
	u32 radii = (n / 2) - 1;
	
	dim3 blocks(radii);
	dim3 threads_per_block(RADIAN_SAMPLES);

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	polar_average_impl<<<blocks,threads_per_block,0,cuda_stream>>>(input_magnitudes, n, output_magnitude_sums, output_frequencies);
}

static __global__ void cuda_copy_cv_mat_impl(cv::cudev::PtrStepSz<f32> cv_mat, f32* device_mem, size_t device_mem_size) {
	const u32 device_mem_index = cv_mat.cols * threadIdx.y + threadIdx.x;
	if (device_mem_index < device_mem_size) {
		device_mem[device_mem_index] = cv_mat.ptr(threadIdx.y)[threadIdx.x];
	}
}

void cuda_copy_cv_mat(const cv::cuda::GpuMat& cv_mat, f32* device_mem, size_t device_mem_size, void* cuda_stream) {
	dim3 threads_per_block(cv_mat.cols, cv_mat.rows);
	cuda_copy_cv_mat_impl<<<1,threads_per_block,0,(cudaStream_t)cuda_stream>>>(cv_mat, device_mem, device_mem_size);
}

static __global__ void cuda_copy_spectrum_value_impl(f32* spectrum_ptr, f32* value) {
	*spectrum_ptr = 1.f - (1.f / (1.f + expf(-3.f * (-(*value) - 2.f))));
}

void cuda_copy_spectrum_value(f32* spectrum_ptr, f32* value, void* cuda_stream) {
	cuda_copy_spectrum_value_impl<<<1,1,0,(cudaStream_t)cuda_stream>>>(spectrum_ptr, value);
}

f32* cuda_cv_mat_ptr(const cv::cuda::GpuMat& cv_mat, u32 x, u32 y) {
	cv::cudev::PtrStepSz<f32> cv_mat_accessor = cv_mat;
	return &cv_mat_accessor.ptr(y)[x];
}
