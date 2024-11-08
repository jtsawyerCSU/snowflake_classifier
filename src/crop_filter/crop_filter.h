#pragma once

#ifndef _crop_filter_H_
#define _crop_filter_H_

#include "crop_filter/core/defines.h"
#include "crop_filter/core/logger.h"
#include "crop_filter/blur_detector.h"
#include "crop_filter/isolator.h"
#include <memory>
#include <utility>
#include <vector>

struct crop_filter {
public:
	
	// a crop_filter object is meant to process 
	// images of the same width and height
	// if you need to change it, make a new object
	// uses stdout as default logging output
	crop_filter(u32 width, u32 height);
	
	// initialize with logging callbacks to redirect any startup messages
	crop_filter(u32 width, u32 height, 
				logger::callback_t normal_logging_callback, 
				logger::callback_t error_logging_callback);
	
	// destructor
	~crop_filter();
	
	// a logger::callback_t logging callback function needs to have signature like
	// e.g. "void foo_log(const char* message)" or
	// e.g. "void foo_log(const char* message, size_t message_length)"
	// for more detail, see logger.h
	
	// used for normal logging to a file or some other stream, default is stdout
	void set_normal_logging_callback(logger::callback_t normal_logging_callback);
	
	// used for error logging to a file or some other stream, default is stderr
	void set_error_logging_callback(logger::callback_t error_logging_callback);
	
	// image blurrines threshold, default is 0.2
	void set_blur_threshold(f32 threshold);
	
	// enable or disable spectral sharpness calculations
	void enable_spectral(bool enable);
	
	// enable or disable spatial sharpness calculations
	void enable_spatial(bool enable);
	
	// enable or disable saving blurry images
	void enable_save_blurry(bool enable);
	
	// enable or disable saving oversize images
	void enable_save_oversize(bool enable);
	
	// returns the number of images above the blur threshold
	u32 crop(const cv::cuda::GpuMat& image);
	
	// returns a vector containing the cropped images above the blur threshold
	const std::vector<cv::Mat>& get_cropped_images();
	
	// returns a vector containing the cropped image coords above the blur threshold
	const std::vector<std::pair<u32, u32>>& get_cropped_coords();
	
	// returns a vector containing the cropped image sharpness above the blur threshold
	const std::vector<f32>& get_cropped_sharpness();
	
	// returns a vector containing the cropped images below the blur threshold
	const std::vector<cv::Mat>& get_blurry_images();
	
	// returns a vector containing the cropped image coords below the blur threshold
	const std::vector<std::pair<u32, u32>>& get_blurry_coords();
	
	// returns a vector containing the cropped image sharpness below the blur threshold
	const std::vector<f32>& get_blurry_sharpness();
	
	// returns a vector containing the cropped images over 300 pixels in some dimension
	const std::vector<cv::Mat>& get_oversize_images();
	
	// returns a vector containing the cropped image coords over 300 pixels in some dimension
	const std::vector<std::pair<u32, u32>>& get_oversize_coords();
	
private:
	
	logger m_logger;
	cv::cuda::Stream m_stream;
	
	std::unique_ptr<isolator> m_snow_isolator;
	std::unique_ptr<blur_detector> m_s3;
	
	std::vector<cv::Mat> m_cropped_images;
	std::vector<std::pair<u32, u32>> m_cropped_coords;
	std::vector<f32> m_cropped_sharpness;
	
	std::vector<cv::Mat> m_blurry_images;
	std::vector<std::pair<u32, u32>> m_blurry_coords;
	std::vector<f32> m_blurry_sharpness;
	
	std::vector<cv::Mat> m_oversize_images;
	std::vector<std::pair<u32, u32>> m_oversize_coords;
	
	cv::cuda::GpuMat m_current_image;
	cv::cuda::GpuMat m_padded_current_image;
	cv::Mat m_current_cpu_image;
	
	f32 m_blur_threshold = 0.2f;
	u32 m_width = 0;
	u32 m_height = 0;
	bool m_spectral_enabled = true;
	bool m_spatial_enabled = true;
	bool m_blurry_enabled = true;
	bool m_oversize_enabled = true;
	
};

#endif
