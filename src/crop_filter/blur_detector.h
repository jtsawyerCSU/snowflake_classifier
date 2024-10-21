#pragma once

#ifndef _blur_detector_H
#define _blur_detector_H

#include "crop_filter/core/defines.h"
#include "crop_filter/cuda/cuda_polyfit.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/cudaarithm.hpp>
#include <memory>

struct blur_detector {
public:
	
	blur_detector(u32 width, u32 height);
	
	~blur_detector();
	
	void resize(u32 width, u32 height);
	
	f32 find_S3_max(const cv::cuda::GpuMat& image);
	
	u32 m_image_width = 0;
	u32 m_image_height = 0;
	
private:
	
	void generate_spectral_map(const cv::cuda::GpuMat& image);
	
	void generate_spatial_map(const cv::cuda::GpuMat& img);
	
	void amplitude_spectrum_slope(const cv::cuda::GpuMat& block, f32* spectral_map_ptr);
	
	cv::cuda::Stream m_stream;
	cv::cuda::GpuMat m_spectral_map;
	cv::cuda::GpuMat m_spatial_map;
	cv::cuda::GpuMat m_padded_input_image;
	cv::Size m_map_size;
	u32 m_padding_x = 0;
	u32 m_padding_y = 0;
	
	// spectral map members
	/////////////////////////////////////////
	cv::cuda::GpuMat m_padded_spectral_image;
	cv::cuda::GpuMat m_spectral_luminance_map;
	cv::cuda::GpuMat m_spectral_slope_map;
	cv::cuda::GpuMat m_hanning_window;
	cv::cuda::GpuMat m_block_window_product;
	cv::cuda::GpuMat m_combined_complex;
	cv::cuda::GpuMat m_split_complex[2];
	cv::cuda::GpuMat m_magnitudes_gpu;
	cv::cuda::GpuMat m_frequencies_gpu;
	cv::cuda::GpuMat m_magnitude_sums_gpu;
	
	cv::Size m_padded_spectral_image_size;
	cv::Size m_internal_spectral_map_size;
	
	cv::Rect kernel;
	
	cv::Ptr<cv::cuda::DFT> m_cuda_dft;
	
	std::unique_ptr<cuda_polyfit> m_gpu_solver;
	/////////////////////////////////////////
	
	// spatial map members
	/////////////////////////////////////////
	cv::cuda::GpuMat m_partial_spatial_map1;
	cv::cuda::GpuMat m_partial_spatial_map2;
	cv::cuda::GpuMat m_intermidiate_map1;
	cv::cuda::GpuMat m_intermidiate_map2;
	cv::cuda::GpuMat m_padded_spatial_image;
	
	cv::Rect m_intermidiate_map2_adjust_box;
	
	cv::Size m_padded_spatial_image_size;
	cv::Size m_partial_spatial_map1_size;
	cv::Size m_partial_spatial_map2_size;
	cv::Size m_intermidiate_map1_size;
	cv::Size m_intermidiate_map2_size;
	/////////////////////////////////////////
	
};

#endif