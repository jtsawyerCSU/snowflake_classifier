#pragma once

#ifndef _crop_filter_H_
#define _crop_filter_H_

#include "crop_filter/core/defines.h"
#include "crop_filter/core/logger.h"
#include "crop_filter/blur_detector.h"
#include "crop_filter/Isolator.h"
#include <memory>
#include <utility>
#include <span>
#include <vector>

struct crop_filter {
public:
	
	// a crop_filter object is meant to process 
	// images of the same width and height
	// if you need to change it, make a new object
	// uses stdout as default logging output
	crop_filter(u32 width, u32 height);
	
	// initialize with a logging callback to redirect any startup messages
	crop_filter(u32 width, u32 height, std::function<void (const char*)> logging_callback);
	crop_filter(u32 width, u32 height, std::function<void (const char*, size_t)> logging_callback);
	
	// destructor
	~crop_filter();
	
	// used for logging to a file or some other stream, default is stdout
	// a logging callback function needs to have signature like
	// e.g. "void foo_log(const char* message)" or
	// e.g. "void foo_log(const char* message, size_t message_length)"
	void set_logging_callback(std::function<void (const char*)> logging_callback);
	void set_logging_callback(std::function<void (const char*, size_t)> logging_callback);
	
	// image blurrines threshold, default is 0.2
	void set_blur_threshold(f32 threshold);
	
	// returns the number of images above the blur threshold
	u32 crop(const cv::cuda::GpuMat& image);
	
	// for the unfamiliar, a span is an immutable view of some memory
	// just treat it like an immutable vector
	
	// returns a span containing the cropped images above the blur threshold
	std::span<cv::cuda::GpuMat> get_cropped_images();
	
	// returns a span containing the cropped image coords above the blur threshold
	std::span<std::pair<u32, u32>> get_cropped_coords();
	
	// returns a span containing the cropped images below the blur threshold
	std::span<cv::cuda::GpuMat> get_blurry_images();
	
	// returns a span containing the cropped image coords below the blur threshold
	std::span<std::pair<u32, u32>> get_blurry_coords();
	
private:
	
	std::unique_ptr<Isolator> m_snow_isolator;
	std::unique_ptr<blur_detector> m_s3;
	
	std::vector<cv::cuda::GpuMat> m_cropped_images;
	std::vector<std::pair<u32, u32>> m_cropped_coords;
	
	std::vector<cv::cuda::GpuMat> m_blurry_images;
	std::vector<std::pair<u32, u32>> m_blurry_coords;
	
	f32 m_blur_threshold = 0.2f;
	u32 m_width = 0;
	u32 m_height = 0;
	
	logger m_logger;
	
	// S3 members
	/////////////////////////////////////////
	cv::cuda::GpuMat m_byte_image;
	
	/////////////////////////////////////////
	
};

#endif
