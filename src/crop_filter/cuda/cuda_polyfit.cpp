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

cuda_polyfit::cuda_polyfit(u32 points, logger& logger) : 
	m_logger(logger) {
	
    m_internal_state = std::make_unique<opaque_cuda_state>();
    
	m_internal_state->cusolver_status = cusolverDnCreate(&m_internal_state->cusolver_handle);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error while creating cusolver context!");
	
	m_points = points;
	m_width = points;
	m_height = 2; // cuda_polyfit is only capable of solving for first order polynomials
	
	// create the params and info structure for the 'expert' interface
	m_internal_state->cusolver_status = cusolverDnIRSParamsCreate(&m_internal_state->gels_irs_params);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Failed to create cusolver params structure!");
	m_internal_state->cusolver_status = cusolverDnIRSInfosCreate(&m_internal_state->gels_irs_infos);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Failed to create cusolver info structure!");
    
	// Set the main and the low precision of the solver DSgels 
	m_internal_state->cusolver_status = cusolverDnIRSParamsSetSolverPrecisions(m_internal_state->gels_irs_params, CUSOLVER_R_32F, CUSOLVER_R_32F);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error while setting cusolver precisions!");
    
	// Set the solver refinement.
	m_internal_state->cusolver_status = cusolverDnIRSParamsSetRefinementSolver(m_internal_state->gels_irs_params, CUSOLVER_IRS_REFINE_NONE);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error while setting cusolver refinment mode!");
	
	// Get work buffer size
	m_internal_state->cusolver_status = cusolverDnIRSXgels_bufferSize(m_internal_state->cusolver_handle, m_internal_state->gels_irs_params, m_width, m_height, 1, &m_workspace_size);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error while calculating required cusolver gels workspace size!");
	
	// Allocate workspace
	m_internal_state->cuda_status = cudaMalloc((void**)&m_device_workspace, m_workspace_size);
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while allocating memory with CUDA!");
    
	// Allocate space in the GPU
	m_internal_state->cuda_status = cudaMalloc((void**)&m_device_A, sizeof(f32) * m_width * m_height);
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while allocating memory with CUDA!");
	m_internal_state->cuda_status = cudaMalloc((void**)&m_device_B, sizeof(f32) * m_width);
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while allocating memory with CUDA!");
	m_internal_state->cuda_status = cudaMalloc((void**)&m_device_X, sizeof(f32) * m_height);
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while allocating memory with CUDA!");
	m_internal_state->cuda_status = cudaMalloc((void**)&m_device_info, sizeof(s32));
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while allocating memory with CUDA!");
    
    m_A_default = std::vector<f32>(m_width * m_height);
    std::fill_n(m_A_default.data(), m_A_default.size(), 1.f);
    m_internal_state->cuda_status = cudaMemcpy(m_device_A, m_A_default.data(), sizeof(f32) * m_A_default.size(), cudaMemcpyHostToDevice);
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while copying memory to CUDA device!");
}

cuda_polyfit::~cuda_polyfit() {
    ASSERT_MSG(m_logger, m_internal_state != nullptr, "cuda_polyfit was somehow initialized without initalizing cuda_polyfit! This should be impossible!");
    
	if (m_device_A != nullptr) {
        m_internal_state->cuda_status = cudaFree(m_device_A);
	    ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (m_device_B != nullptr) {
        m_internal_state->cuda_status = cudaFree(m_device_B);
	    ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (m_device_X != nullptr) {
        m_internal_state->cuda_status = cudaFree(m_device_X);
	    ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (m_device_info != nullptr) {
        m_internal_state->cuda_status = cudaFree(m_device_info);
	    ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
	if (m_device_workspace != nullptr) {
        m_internal_state->cuda_status = cudaFree(m_device_workspace);
	    ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while deallocating memory with CUDA!");
    }
    
	m_internal_state->cusolver_status = cusolverDnIRSParamsDestroy(m_internal_state->gels_irs_params);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error while destroying cusolver params structure!");
    
	m_internal_state->cusolver_status = cusolverDnIRSInfosDestroy(m_internal_state->gels_irs_infos);
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error, cusolver info structure must be initialized before being destroyed!");
    
	if (m_internal_state->cusolver_handle != nullptr) {
        m_internal_state->cusolver_status = cusolverDnDestroy(m_internal_state->cusolver_handle);
	    ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error, cusolver context must be initialized before being destroyed!");
    }
}
    
void cuda_polyfit::set_stream(cv::cuda::Stream& stream) {
    m_internal_state->cuda_stream = cv::cuda::StreamAccessor::getStream(stream);
	m_internal_state->cusolver_status = cusolverDnSetStream(m_internal_state->cusolver_handle, m_internal_state->cuda_stream);
}

void cuda_polyfit::solve(const cv::cuda::GpuMat& src_x,
                    	 const cv::cuda::GpuMat& src_y,
                         f32* device_output_pointer) {
	ASSERT_MSG(m_logger, (u32)src_x.rows == m_width, "src_x.rows must be the same size as points cuda_polyfit was initialized with!");
	ASSERT_MSG(m_logger, (u32)src_y.rows == m_width, "src_y.rows must be the same size as points cuda_polyfit was initialized with!");
	ASSERT_MSG(m_logger, (src_x.cols == 1) && (src_y.cols == 1), "src_x and src_y must be single column Mats!");
    
    // reset m_device_A
    m_internal_state->cuda_status = cudaMemcpyAsync(m_device_A, m_A_default.data(), sizeof(f32) * m_width, cudaMemcpyHostToDevice, m_internal_state->cuda_stream);
	ASSERT_MSG(m_logger, cudaSuccess == m_internal_state->cuda_status, "Error while copying memory to CUDA device!");
    
	// Copy matrices from opencv format to raw cuda format
    cuda_copy_cv_mat(src_x, m_device_A + m_width, m_width, m_internal_state->cuda_stream);
    cuda_copy_cv_mat(src_y, m_device_B, m_width, m_internal_state->cuda_stream);
	
	s32 number_iterations = 0;
    s32 max_wh = std::max(m_width, m_height);
    
	// run (cu)solver
	m_internal_state->cusolver_status = cusolverDnIRSXgels(m_internal_state->cusolver_handle,
									                     m_internal_state->gels_irs_params,
									                     m_internal_state->gels_irs_infos,
									                     m_width,
									                     m_height,
									                     1,
									                     (void*)m_device_A,
									                     m_width,
									                     (void*)m_device_B,
									                     max_wh,
									                     (void*)m_device_X,
									                     max_wh,
									                     m_device_workspace,
									                     m_workspace_size,
									                     &number_iterations,
									                     m_device_info);
    
	ASSERT_MSG(m_logger, CUSOLVER_STATUS_SUCCESS == m_internal_state->cusolver_status, "Error while solving in cuda_polyfit!");
	
    // ignore gpu info and number iterations
    // I haven't thought of any very good asyncronous way to check them
    // s32 info_gpu = 0;
	// m_internal_state->cuda_status = cudaMemcpyAsync(&info_gpu, m_device_info, sizeof(int), cudaMemcpyDeviceToHost, m_internal_state->cuda_stream);
	// assert(cudaSuccess == m_internal_state->cuda_status);
	// //printf("after gels: info_gpu = %d\n", info_gpu);
	// //printf("after gels: number_iterations = %d\n", number_iterations);
	
    cuda_copy_spectrum_value(device_output_pointer, &m_device_X[0], m_internal_state->cuda_stream);
}
