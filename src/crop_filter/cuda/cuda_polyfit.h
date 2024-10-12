#pragma once

#ifndef _cuda_polyfit_H_
#define _cuda_polyfit_H_

#include "crop_filter/core/defines.h"
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <vector>

struct opaque_cuda_state;

struct cuda_polyfit {
    
	cuda_polyfit(u32 cols, u32 rows);
	~cuda_polyfit();
    
    void set_stream(cv::cuda::Stream& stream);
    
	int solve(const cv::cuda::GpuMat& src_x,
              const cv::cuda::GpuMat& src_y,
			  f32* device_output_pointer);

private:
	
    std::unique_ptr<opaque_cuda_state> internal_state;
    std::vector<float> A_default;
    
	u32 width = 0;
	u32 height = 0;
	size_t workspace_size = 0;
	
	f32* device_workspace = nullptr;
	f32* device_A = nullptr;
	f32* device_B = nullptr;
	f32* device_X = nullptr;
	s32* device_info = nullptr;
    
};

#endif
