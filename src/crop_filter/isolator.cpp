//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion" // opencv uses enum conversion a lot. the warnings are distracting

#include "crop_filter/isolator.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <algorithm>
#include <opencv2/cudawarping.hpp>
#include "util/savepng.h"

isolator::isolator(logger& logger, cv::cuda::Stream& stream) : 
	m_logger(logger),
	m_stream(stream) {}

constexpr f64 scalar = 8;
constexpr f64 inverse_scalar = 1 / scalar;
constexpr s32 point_expand_factor = 5;
constexpr s32 box_expand_factor = 20;

u32 isolator::isolate_flakes(const cv::cuda::GpuMat& input) {
	
	// fill averaging background buffer
	if (m_needs_background) {
		// image must be unsigned 8-bit
		input.convertTo(m_backgrounds[m_last_background_index], CV_8U, m_stream);
		
		// divide stored background by 4
		cv::cuda::rshift(m_backgrounds[m_last_background_index], cv::Scalar(2), m_backgrounds[m_last_background_index], m_stream);
		
		// if 4 images have been collected, add them together
		if (++m_last_background_index >= 4) {
			--m_last_background_index;
			cv::cuda::add(m_backgrounds[0], m_backgrounds[1],      m_averaged_background, {}, {}, m_stream);
			cv::cuda::add(m_backgrounds[2], m_averaged_background, m_averaged_background, {}, {}, m_stream);
			cv::cuda::add(m_backgrounds[3], m_averaged_background, m_averaged_background, {}, {}, m_stream);
			m_needs_background = false;
		}
		
		// wait for stream to complete
		m_stream.waitForCompletion();
		
		return 0;
	}
	
	// image must be unsigned 8-bit
	input.convertTo(m_foreground, CV_8U, m_stream);
	
	// subtract the averaged background from the foreground image
	cv::cuda::subtract(m_foreground, m_averaged_background, m_intermediate_image2, {}, {}, m_stream);
	
	// downscale the subtracted image using linear interpolation
	cv::cuda::resize(m_intermediate_image2, m_intermediate_image1, {}, inverse_scalar, inverse_scalar, cv::INTER_LINEAR, m_stream);
	
	// threshold the downscaled image
	cv::cuda::threshold(m_intermediate_image1, m_intermediate_image, 30, 255, cv::THRESH_BINARY, m_stream);
	
	// download the thresholded image to the cpu
	m_intermediate_image.download(m_cpu_image, m_stream);
	
	// threshold the background subtracted image to suppress border noise
	cv::cuda::threshold(m_intermediate_image2, m_intermediate_image2, 5, 255, cv::THRESH_BINARY, m_stream);
	
	// bitwise OR the foreground image with itself using the background subtracted image as a mask
	// the bitwise OR does nothing, this just masks out the background 
	cv::cuda::bitwise_or(m_foreground, m_foreground, m_intermediate_image1, m_intermediate_image2, m_stream);
	
	// wait for stream to complete
	m_stream.waitForCompletion();
	
	// add new image to background
	if (++m_last_background_index >= 4) {
		m_last_background_index = 0;
	}
	// subtract the oldest background from the averaged background
	cv::cuda::subtract(m_averaged_background, m_backgrounds[m_last_background_index], m_averaged_background, {}, {}, m_stream);
	// store the current foreground into the backgrounds array
	std::swap(m_backgrounds[m_last_background_index], m_foreground);
	// divide the new background by 4
	cv::cuda::rshift(m_backgrounds[m_last_background_index], cv::Scalar(2), m_backgrounds[m_last_background_index], m_stream);
	// add the new background to the average
	cv::cuda::add(m_backgrounds[m_last_background_index], m_averaged_background, m_averaged_background, {}, {}, m_stream);
	
	// find non-masked pixels where snowflakes might be
	cv::findNonZero(m_cpu_image, m_points);
	
	// combine points found into blobs to identify likely snowflakes
	get_bounded_images(15);
	
	// wait for m_stream to complete
	m_stream.waitForCompletion();
	
	return m_output_flakes.size();
}

void isolator::get_bounded_images(u32 minimum_dimensions) {
	// clear output from last iteration
	m_output_flakes.clear();
	m_output_coords.clear();
	
	// if no points were found, leave
	if (m_points.empty()) {
		return;
	}
	
	// create bounding boxes from points
	m_boxes.reserve(m_points.size());
	for (const cv::Point& point : m_points) {
		m_boxes.emplace_back(point);
	}
	m_points.clear();
	
	// combine overlapping boxes
loop:
	for (s32 i = 0; i < ((s32)m_boxes.size() - 1); ++i) {
		bounding_box& combinedBox = m_boxes[i];
		std::vector<bounding_box>::iterator it = std::remove_if(m_boxes.begin() + i + 1, m_boxes.end(), 
			[&](const bounding_box& box) {
				if (combinedBox == box) {
					combinedBox.combine(box);
					return true;
				} else {
					return false;
				}
			});
		if (it != m_boxes.end()) {
			m_boxes.erase(it, m_boxes.end());
			goto loop;
		}
	}
	
	// resize output vectors to accept at all bounded images
	m_output_flakes.resize(m_boxes.size());
	m_output_coords.resize(m_boxes.size());
	
	u32 output_index = 0;
	for (const bounding_box& combinedBox : m_boxes) {
		cv::Rect bounding_rectangle = combinedBox;
		
		// scale up bounding_rectangle to the full image size
		bounding_rectangle.x *= scalar;
		bounding_rectangle.y *= scalar;
		bounding_rectangle.width *= scalar;
		bounding_rectangle.height *= scalar;
		
		// if either dimension is less than minimum_dimensions the skip this section
		if (((u32)bounding_rectangle.width < minimum_dimensions) || ((u32)bounding_rectangle.height < minimum_dimensions)) {
			continue;
		}
		
		// expand the bounding rectangle by box_expand_factor
		bounding_rectangle.x -= box_expand_factor;
		bounding_rectangle.y -= box_expand_factor;
		bounding_rectangle.width += box_expand_factor << 1;
		bounding_rectangle.height += box_expand_factor << 1;
		
		// if the bounds of the rectangle extend beyond the edges of the image, skip this section
		if (bounding_rectangle.x < 0 ||
			bounding_rectangle.y < 0 ||
			bounding_rectangle.x + bounding_rectangle.width > m_intermediate_image1.cols ||
			bounding_rectangle.y + bounding_rectangle.height > m_intermediate_image1.rows) {
			continue;
		}
		
		// find the center point of the bounding rectangle
		u32 centerX = bounding_rectangle.x + bounding_rectangle.width / 2;
		u32 centerY = bounding_rectangle.y + bounding_rectangle.height / 2;
		
		m_output_flakes[output_index] = m_intermediate_image1(bounding_rectangle);
		m_output_coords[output_index] = {centerX, centerY};
		
		// increment output index
		++output_index;
	}
	
	// shrink output to final snowflake count
	m_output_flakes.resize(output_index);
	m_output_coords.resize(output_index);
	
	// clear m_boxes in preparation for the next image
	m_boxes.clear();
}

// bounding_box member functions

// bounding_box: constructor of bounding_box from a cv::Point
isolator::bounding_box::bounding_box(const cv::Point& p) : 
	minX{p.x - point_expand_factor}, 
	minY{p.y - point_expand_factor}, 
	maxX{p.x + point_expand_factor}, 
	maxY{p.y + point_expand_factor} {}

// bounding_box: conversion operator to cv::Rect
isolator::bounding_box::operator cv::Rect() const {
	cv::Rect r{};
	r.x = minX;
	r.y = minY;
	r.width = maxX - minX;
	r.height = maxY - minY;
	return r;
}

// bounding_box: returns equal if bounding m_boxes overlap
bool isolator::bounding_box::operator==(const bounding_box& other) const {
	return !((maxX < other.minX) || (minX > other.maxX) || (maxY < other.minY) || (minY > other.maxY));
}

// bounding_box: combines bounding m_boxes using their widest respective bounds
void isolator::bounding_box::combine(const bounding_box& other) {
	minX = std::min(minX, other.minX);
	maxX = std::max(maxX, other.maxX);
	minY = std::min(minY, other.minY);
	maxY = std::max(maxY, other.maxY);
}
