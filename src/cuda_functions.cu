#include "cuda_functions.h"
#include <opencv2/cudev.hpp>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cmath>
#include "Timer.h"
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <opencv2/cudaarithm.hpp>
#include <cusolverDn.h>

#define SPATIAL_BLOCK_SIZE 8

static __global__ void spatial_map_impl(const cv::cudev::PtrStepSz<float> input, 
										cv::cudev::PtrStepSz<float> output) {
	__shared__ float pixel_variation[(SPATIAL_BLOCK_SIZE - 1) * (SPATIAL_BLOCK_SIZE - 1)];
	const int x = SPATIAL_BLOCK_SIZE * blockIdx.x + threadIdx.x;
	const int y = SPATIAL_BLOCK_SIZE * blockIdx.y + threadIdx.y;

	if ((x < (input.cols - 1)) && (y < (input.rows - 1))) {
		float temp_sum = 0.f;
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
		float max_variation = 0.f;
		for (uint32_t i = 0; i < ((SPATIAL_BLOCK_SIZE - 1) * (SPATIAL_BLOCK_SIZE - 1)); ++i) {
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
	
	//std::printf("CUDA string: %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaDeviceSynchronize();
	
	// Timer t;
	// for (int i = 0; i < 500; ++i) {
	//     spatial_map_impl<<<blocks,threads_per_block>>>(input, output);
	//     cudaDeviceSynchronize();
	//     t.lap();
	// }

	//t.print();

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	spatial_map_impl<<<blocks,threads_per_block,0,cuda_stream>>>(input, output);
	
	//std::printf("CUDA string: %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaDeviceSynchronize();

}

#define LUMINANCE_BLOCK_SIZE    32
#define LUMINANCE_DELTA_BLOCK   (LUMINANCE_BLOCK_SIZE  / 4)
#define LUMINANCE_OFFSET        (LUMINANCE_DELTA_BLOCK / 2)

static __global__ void luminance_map_impl(const cv::cudev::PtrStepSz<float> input, 
										  cv::cudev::PtrStepSz<float> output) {
	__shared__ float pixel_luminance[LUMINANCE_BLOCK_SIZE * LUMINANCE_BLOCK_SIZE];
	const int x = LUMINANCE_DELTA_BLOCK * blockIdx.x + threadIdx.x;
	const int y = LUMINANCE_DELTA_BLOCK * blockIdx.y + threadIdx.y;

	float pixel = input.ptr(y + LUMINANCE_OFFSET)[x + LUMINANCE_OFFSET];
	pixel *= 0.0364f;
	pixel += 0.7656f;
	pixel = pow(pixel, 2.2f);
	pixel_luminance[threadIdx.x + threadIdx.y * LUMINANCE_BLOCK_SIZE] = pixel;

	__syncthreads();
	if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
		float max_luminance = pixel_luminance[0];
		float min_luminance = pixel_luminance[0];
		float luminance_sum = 0.f;
		for (uint32_t i = 0; i < (LUMINANCE_BLOCK_SIZE * LUMINANCE_BLOCK_SIZE); ++i) {
			if (max_luminance < pixel_luminance[i]) {
				max_luminance = pixel_luminance[i];
			}
			if (min_luminance > pixel_luminance[i]) {
				min_luminance = pixel_luminance[i];
			}
			luminance_sum += pixel_luminance[i];
		}
		float mean_luminance = luminance_sum / (LUMINANCE_BLOCK_SIZE * LUMINANCE_BLOCK_SIZE);
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
	
	//std::printf("CUDA string: %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaDeviceSynchronize();

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	luminance_map_impl<<<blocks,threads_per_block,0,cuda_stream>>>(input, output);
	
	//std::printf("CUDA string: %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaDeviceSynchronize();

}

#define RADIAN_STEPS 360
#define RADIAN_SAMPLES (RADIAN_STEPS - 1)

static __global__ void polar_average_impl(const cv::cudev::PtrStepSz<float> input_magnitudes, 
										  uint32_t input_n, 
										  cv::cudev::PtrStepSz<float> output_magnitude_sums, 
										  cv::cudev::PtrStepSz<float> output_frequencies) {
	const float two_pi = atanf(1.f) * 8.f;

	__shared__ float magnitude_at_angles[RADIAN_SAMPLES];

	const uint32_t radius   = blockIdx.x + 1;
	const uint32_t angle    = threadIdx.x * (two_pi / RADIAN_STEPS);

	float x = radius * sinf(angle); // x coordinate
	float y = radius * cosf(angle); // y coordinate
	
	float x1 = std::copysign(std::floor(std::abs(x)), x); // aliasing
	float x2 = std::copysign(std::ceil(std::abs(x)), x);
	float y1 = std::copysign(std::floor(std::abs(y)), y);
	float y2 = std::copysign(std::ceil(std::abs(y)), y);
	
	float ex = std::abs(x - x1);
	float ey = std::abs(y - y1);
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
	
	float f11 = input_magnitudes.ptr((uint32_t)y1)[(uint32_t)x1];
	float f12 = input_magnitudes.ptr((uint32_t)y2)[(uint32_t)x1];
	float f21 = input_magnitudes.ptr((uint32_t)y1)[(uint32_t)x2];
	float f22 = input_magnitudes.ptr((uint32_t)y2)[(uint32_t)x2];
	
	magnitude_at_angles[threadIdx.x] = (f21 - f11) * ex * (1.f - ey) + (f12 - f11) * (1.f - ex) * ey + (f22 - f11) * ex * ey + f11;

	__syncthreads();
	if (threadIdx.x == 0) {
		float magnitude_at_radius = 0.f;
		for (uint32_t i = 0; i < RADIAN_SAMPLES; ++i) {
			magnitude_at_radius += magnitude_at_angles[i];
		}
		magnitude_at_radius /= RADIAN_STEPS;
		output_magnitude_sums.ptr(blockIdx.x)[0] = magnitude_at_radius;
		output_frequencies.ptr(blockIdx.x)[0] = (0.5f / (input_n / 2)) * (blockIdx.x + 1);
	}
}

void cuda_polar_average(const cv::cuda::GpuMat& input_magnitudes, cv::cuda::GpuMat& output_magnitude_sums, cv::cuda::GpuMat& output_frequencies, cv::cuda::Stream& stream) {
	uint32_t n = std::min(input_magnitudes.rows, input_magnitudes.cols);
	uint32_t radii = (n / 2) - 1;
	
	dim3 blocks(radii);
	dim3 threads_per_block(RADIAN_SAMPLES);

	//std::printf("CUDA string: %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaDeviceSynchronize();
	
	output_magnitude_sums = cv::cuda::GpuMat(radii, 1, CV_32F);
	output_frequencies = cv::cuda::GpuMat(radii, 1, CV_32F);

	cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	polar_average_impl<<<blocks,threads_per_block,0,cuda_stream>>>(input_magnitudes, n, output_magnitude_sums, output_frequencies);
	
	//std::printf("CUDA string: %s\n", cudaGetErrorString(cudaGetLastError()));
	//cudaDeviceSynchronize();
}

static __global__ void make_mat_continuous_impl(cv::cudev::PtrStepSz<float> mat, float* device_mem, size_t device_mem_size) {
	const int index = mat.cols * blockIdx.y + blockIdx.x;
	if (index < device_mem_size) {
		device_mem[index] = mat.ptr(blockIdx.y)[blockIdx.x];
	}
}

static __global__ void insert_spectrum_value_impl(cv::cudev::PtrStepSz<float> spectral_map, 
												  size_t block_x, size_t block_y,
												  float* spectrum_poly_slope) {
	// spectrum_slope is the slope of power spectrum of the block
	// Input to a sigmoid function
	spectral_map.ptr(block_y)[block_x] = 1.f - (1.f / (1.f + expf(-3.f * (-spectrum_poly_slope[0] - 2.f))));
}

#ifndef max
#define max(a, b) ((a) > (b) ? (a) : (b))
#endif
struct cuda_polyfit {
private:
	cusolverDnHandle_t cusolver_handle = nullptr;
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cuda_status = cudaSuccess;
	cusolverDnIRSParams_t gels_irs_params = nullptr;
	cusolverDnIRSInfos_t gels_irs_infos = nullptr;
	
	int m = 0;
	int n = 0;
	int lda = 0;
	int ldb = 0;
	int ldx = 0;
	int nrhs = 0;
	size_t lwork = 0;
	int niter = 0;
	int info_gpu = 0;
	
	float* d_A = nullptr;
	float* d_B = nullptr;
	float* d_X = nullptr;
	float* d_work = nullptr;
	int* d_info = nullptr;
	
public:
	cuda_polyfit(int cols, int rows) {
		cusolver_status = cusolverDnCreate(&cusolver_handle);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		
		m = cols;
		n = rows;
		lda = m;
		ldb = max(m, n);
		ldx = max(m, n);
		nrhs = 1;
		lwork = 0;
		niter = 0;
		info_gpu = 0;
		
		// Allocate space in the GPU
		cuda_status = cudaMalloc((void**)&d_A, sizeof(float) * m * n);
		assert(cudaSuccess == cuda_status);
		cuda_status = cudaMalloc((void**)&d_B, sizeof(float) * m * nrhs);
		assert(cudaSuccess == cuda_status);
		cuda_status = cudaMalloc((void**)&d_X, sizeof(float) * n * nrhs);
		assert(cudaSuccess == cuda_status);
		cuda_status = cudaMalloc((void**)&d_info, sizeof(int));
		assert(cudaSuccess == cuda_status);
		
		// create the params and info structure for the expert interface
		cusolverDnIRSParamsCreate(&gels_irs_params);
		cusolverDnIRSInfosCreate(&gels_irs_infos);
		// Set the main and the low precision of the solver DSgels 
		cusolver_status = cusolverDnIRSParamsSetSolverPrecisions(gels_irs_params, CUSOLVER_R_32F, CUSOLVER_R_32F);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		// Set the refinement solver.
		cusolver_status = cusolverDnIRSParamsSetRefinementSolver(gels_irs_params, CUSOLVER_IRS_REFINE_NONE);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		
		// Get work buffer size
		cusolver_status = cusolverDnIRSXgels_bufferSize(cusolver_handle, gels_irs_params, m, n, nrhs, &lwork);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		
		// Allocate workspace
		cuda_status = cudaMalloc((void**)&d_work, lwork);
		assert(cudaSuccess == cuda_status);
	}
	~cuda_polyfit() {
		if (d_A) cudaFree(d_A);
		if (d_B) cudaFree(d_B);
		if (d_X) cudaFree(d_X);
		if (d_info) cudaFree(d_info);
		if (d_work) cudaFree(d_work);
		cusolver_status = cusolverDnIRSParamsDestroy(gels_irs_params);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		cusolver_status = cusolverDnIRSInfosDestroy(gels_irs_infos);
		assert(CUSOLVER_STATUS_SUCCESS == cusolver_status);
		if (cusolver_handle) cusolverDnDestroy(cusolver_handle);
	}
	int solve(const cv::cuda::GpuMat& src_x, const cv::cuda::GpuMat& src_y,
			  cv::cuda::GpuMat& spectral_map, size_t block_x, size_t block_y,
			  cv::cuda::Stream& stream) {
		const int order = 1;
		cv::cuda::GpuMat M(src_x.rows, order + 1, CV_32FC1);
		cv::cuda::GpuMat copy;
		for (size_t i = 0; i <= order; i++) {
			copy = src_x.clone();
			cv::cuda::pow(copy, i, copy, stream);
			cv::cuda::GpuMat M1 = M.col(i);
			copy.col(0).copyTo(M1, stream);
		}
		cv::cuda::GpuMat M_t;
		cv::cuda::transpose(M, M_t, stream);
		
		cudaStream_t cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
		cusolver_status = cusolverDnSetStream(cusolver_handle, cuda_stream);

		// Copy matrices into GPU space
		dim3 blocks(src_y.cols, src_y.rows);
		dim3 threads_per_block(1, 1);
		make_mat_continuous_impl<<<blocks,threads_per_block,0,cuda_stream>>>(src_y, d_B, sizeof(float) * m * nrhs);
		blocks = dim3(M_t.cols, M_t.rows);
		make_mat_continuous_impl<<<blocks,threads_per_block,0,cuda_stream>>>(M_t, d_A, sizeof(float) * m * n);
		
		// Run solver
		cusolver_status = cusolverDnIRSXgels(cusolver_handle,
											gels_irs_params,
											gels_irs_infos,
											m,
											n,
											nrhs,
											(void*)d_A,
											lda,
											(void*)d_B,
											ldb,
											(void*)d_X,
											ldx,
											d_work,
											lwork,
											&niter,
											d_info);
		//printf("gels status: %d\n", int(cusolver_status));

		// // Copy GPU info
		// cuda_status = cudaMemcpyAsync(&info_gpu, d_info, sizeof(int), cudaMemcpyDeviceToHost, cuda_stream);
		// assert(cudaSuccess == cuda_status);

		// // Get solved data
		// cuda_status = cudaMemcpyAsync(X, d_X, sizeof(float) * n * nrhs, cudaMemcpyDeviceToHost, cuda_stream);
		// assert(cudaSuccess == cuda_status);

		// //printf("after gels: info_gpu = %d\n", info_gpu);
		// //printf("after gels: niter    = %d\n", niter);
		
		insert_spectrum_value_impl<<<1,1,0,cuda_stream>>>(spectral_map, block_x, block_y, (float*)d_X);
		
		return 0;
	}
};

#include <iostream>
void cuda_cv_polyfit(const cv::cuda::GpuMat& src_x, const cv::cuda::GpuMat& src_y, cv::cuda::GpuMat& spectral_map, size_t block_x, size_t block_y, cv::cuda::Stream& stream) {
	static cuda_polyfit gpu_solver(src_x.rows, 2);
	gpu_solver.solve(src_x, src_y, spectral_map, block_x, block_y, stream);
	//stream.waitForCompletion();
}
