#pragma once

#ifndef _cuda_polyfit_H_
#define _cuda_polyfit_H_

#include "crop_filter/core/defines.h"
#include <memory>
#include <opencv2/core/cuda.hpp>
#include <vector>

struct opaque_cuda_state;

// this object uses cusolver to perfrom a 
// linear fit on several points
struct cuda_polyfit {
public:
    
	// points is the number of points that will be input into solve()
	cuda_polyfit(u32 points);
	~cuda_polyfit();
    
	// sets the cuda stream the next call to the solve() uses
    void set_stream(cv::cuda::Stream& stream);
    
	// performs the linear fit and stores the resulting
	// slope at the provided device pointer after 
	// putting it through the sigmoid function
	// specified in the S3 paper
	// src_x is the x coords
	// src_y is the y coords
	// device_output_pointer is the cuda device pointer
	void solve(const cv::cuda::GpuMat& src_x,
               const cv::cuda::GpuMat& src_y,
			   f32* device_output_pointer);

	u32 m_points = 0;

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
