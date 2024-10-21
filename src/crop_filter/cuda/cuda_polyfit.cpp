#include "crop_filter/cuda/cuda_polyfit.h"
#include "crop_filter/cuda/cuda_functions.h"
#include "crop_filter/core/asserts.h"
#include "crop_filter/core/logger.h"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <opencv2/core/cuda_stream_accessor.hpp>
#include <vector>
#include <algorithm>

struct opaque_cuda_state {
	cusolverDnHandle_t cusolver_handle = nullptr;
	cusolverDnIRSParams_t gels_irs_params = nullptr;
	cusolverDnIRSInfos_t gels_irs_infos = nullptr;
    cudaStream_t cuda_stream = nullptr;
    
	cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
	cudaError_t cuda_status = cudaSuccess;
};

cuda_polyfit::cuda_polyfit(u32 points) {
    internal_state = std::make_unique<opaque_cuda_state>();
    
	internal_state->cusolver_status = cusolverDnCreate(&internal_state->cusolver_handle);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error while creating cusolver context!");
	
	m_points = points;
	width = points;
	height = 2; // cuda_polyfit is only capable of solving for first order polynomials
	
	// create the params and info structure for the 'expert' interface
	internal_state->cusolver_status = cusolverDnIRSParamsCreate(&internal_state->gels_irs_params);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Failed to create cusolver params structure!");
	internal_state->cusolver_status = cusolverDnIRSInfosCreate(&internal_state->gels_irs_infos);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Failed to create cusolver info structure!");
    
	// Set the main and the low precision of the solver DSgels 
	internal_state->cusolver_status = cusolverDnIRSParamsSetSolverPrecisions(internal_state->gels_irs_params, CUSOLVER_R_32F, CUSOLVER_R_32F);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error while setting cusolver precisions!");
    
	// Set the solver refinement.
	internal_state->cusolver_status = cusolverDnIRSParamsSetRefinementSolver(internal_state->gels_irs_params, CUSOLVER_IRS_REFINE_NONE);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error while setting cusolver refinment mode!");
	
	// Get work buffer size
	internal_state->cusolver_status = cusolverDnIRSXgels_bufferSize(internal_state->cusolver_handle, internal_state->gels_irs_params, width, height, 1, &workspace_size);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error while calculating required cusolver gels workspace size!");
	
	// Allocate workspace
	internal_state->cuda_status = cudaMalloc((void**)&device_workspace, workspace_size);
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while allocating memory with CUDA!");
    
	// Allocate space in the GPU
	internal_state->cuda_status = cudaMalloc((void**)&device_A, sizeof(f32) * width * height);
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while allocating memory with CUDA!");
	internal_state->cuda_status = cudaMalloc((void**)&device_B, sizeof(f32) * width);
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while allocating memory with CUDA!");
	internal_state->cuda_status = cudaMalloc((void**)&device_X, sizeof(f32) * height);
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while allocating memory with CUDA!");
	internal_state->cuda_status = cudaMalloc((void**)&device_info, sizeof(s32));
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while allocating memory with CUDA!");
    
    A_default = std::vector<f32>(width * height);
    std::fill_n(A_default.data(), A_default.size(), 1.f);
    internal_state->cuda_status = cudaMemcpy(device_A, A_default.data(), sizeof(f32) * A_default.size(), cudaMemcpyHostToDevice);
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while copying memory to CUDA device!");
}

cuda_polyfit::~cuda_polyfit() {
    ASSERT_MSG(internal_state != nullptr, "cuda_polyfit was somehow initialized without initalizing cuda_polyfit! This should be impossible!");
    
	if (device_A != nullptr) {
        internal_state->cuda_status = cudaFree(device_A);
	    ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (device_B != nullptr) {
        internal_state->cuda_status = cudaFree(device_B);
	    ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (device_X != nullptr) {
        internal_state->cuda_status = cudaFree(device_X);
	    ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (device_info != nullptr) {
        internal_state->cuda_status = cudaFree(device_info);
	    ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (device_workspace != nullptr) {
        internal_state->cuda_status = cudaFree(device_workspace);
	    ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
    
	internal_state->cusolver_status = cusolverDnIRSParamsDestroy(internal_state->gels_irs_params);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error while destroying cusolver params structure!");
    
	internal_state->cusolver_status = cusolverDnIRSInfosDestroy(internal_state->gels_irs_infos);
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error, cusolver info structure must be initialized before being destroyed!");
    
	if (internal_state->cusolver_handle != nullptr) {
        internal_state->cusolver_status = cusolverDnDestroy(internal_state->cusolver_handle);
	    ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error, cusolver context must be initialized before being destroyed!");
    }
}
    
void cuda_polyfit::set_stream(cv::cuda::Stream& stream) {
    internal_state->cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	internal_state->cusolver_status = cusolverDnSetStream(internal_state->cusolver_handle, internal_state->cuda_stream);
}

void cuda_polyfit::solve(const cv::cuda::GpuMat& src_x,
                    	 const cv::cuda::GpuMat& src_y,
                         f32* device_output_pointer) {
	ASSERT_MSG((u32)src_x.rows == width, "src_x.rows must be the same size as points cuda_polyfit was initialized with!");
	ASSERT_MSG((u32)src_y.rows == width, "src_y.rows must be the same size as points cuda_polyfit was initialized with!");
	ASSERT_MSG((src_x.cols == 1) && (src_y.cols == 1), "src_x and src_y must be single column Mats!");
    
    // reset device_A
    internal_state->cuda_status = cudaMemcpyAsync(device_A, A_default.data(), sizeof(f32) * width, cudaMemcpyHostToDevice, internal_state->cuda_stream);
	ASSERT_MSG(cudaSuccess == internal_state->cuda_status, "Error while copying memory to CUDA device!");
    
	// Copy matrices from opencv format to raw cuda format
    cuda_copy_cv_mat(src_x, device_A + width, width, internal_state->cuda_stream);
    cuda_copy_cv_mat(src_y, device_B, width, internal_state->cuda_stream);
	
	s32 number_iterations = 0;
    s32 max_wh = std::max(width, height);
    
	// run (cu)solver
	internal_state->cusolver_status = cusolverDnIRSXgels(internal_state->cusolver_handle,
									                     internal_state->gels_irs_params,
									                     internal_state->gels_irs_infos,
									                     width,
									                     height,
									                     1,
									                     (void*)device_A,
									                     width,
									                     (void*)device_B,
									                     max_wh,
									                     (void*)device_X,
									                     max_wh,
									                     device_workspace,
									                     workspace_size,
									                     &number_iterations,
									                     device_info);
    
	ASSERT_MSG(CUSOLVER_STATUS_SUCCESS == internal_state->cusolver_status, "Error while solving in cuda_polyfit!");
	
    // ignore gpu info and number iterations
    // I haven't thought of any very good asyncronous way to check them
    // s32 info_gpu = 0;
	// internal_state->cuda_status = cudaMemcpyAsync(&info_gpu, device_info, sizeof(int), cudaMemcpyDeviceToHost, internal_state->cuda_stream);
	// assert(cudaSuccess == internal_state->cuda_status);
	// //printf("after gels: info_gpu = %d\n", info_gpu);
	// //printf("after gels: number_iterations = %d\n", number_iterations);
	
    cuda_copy_spectrum_value(device_output_pointer, &device_X[0], internal_state->cuda_stream);
}
