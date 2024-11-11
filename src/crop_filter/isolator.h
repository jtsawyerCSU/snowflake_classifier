#pragma once

#ifndef _isolator_H
#define _isolator_H

#include "crop_filter/core/defines.h"
#include "crop_filter/core/logger.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>
#include <vector>
#include <array>

struct isolator {
	
	// logger is the logger instance used for output
	// stream is the opencv cuda stream instance used
	isolator(logger& logger, cv::cuda::Stream& stream);
	
	// returns the number of snowflakes identified
	u32 isolate_flakes(const cv::cuda::GpuMat& input);
	
private:
	
	// helper struct to assist with collating points
	struct bounding_box {
		// some defaulted constructors
		bounding_box() = default;
		bounding_box(const bounding_box&) = default;
		// constructor of bounding_box from a cv::Point
		bounding_box(const cv::Point& p);
		// conversion operator to cv::Rect
		operator cv::Rect() const;
		// returns equal if bounding boxes overlap
		bool operator==(const bounding_box& other) const;
		// combines bounding boxes using their widest respective bounds
		void combine(const bounding_box& other);
		s32 minX{}, minY{}, maxX{}, maxY{};
	};
	
	// combines points found into blobs to identify likely snowflakes
	void get_bounded_images(u32 minimum_dimensions);
	
	logger& m_logger;
	
	std::vector<cv::Point> m_points;
	std::vector<bounding_box> m_boxes;
	cv::cuda::Stream& m_stream;
	std::array<cv::cuda::GpuMat, 4> m_backgrounds;
	cv::cuda::GpuMat m_averaged_background;
	cv::cuda::GpuMat m_foreground;
	cv::cuda::GpuMat m_intermediate_image;
	cv::cuda::GpuMat m_intermediate_image1;
	cv::cuda::GpuMat m_intermediate_image2;
	cv::Mat m_cpu_image;
	u8 m_last_background_index = 0;
	bool m_needs_background = true;
	
public:
	
	std::vector<cv::cuda::GpuMat> m_output_flakes;
	std::vector<std::pair<u32, u32>> m_output_coords;
	
};

#endif
