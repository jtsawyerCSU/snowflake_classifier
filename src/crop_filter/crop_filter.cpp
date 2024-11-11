#include "crop_filter/crop_filter.h"
#include "crop_filter/core/asserts.h"
#include <iostream>

static void default_normal_logger(const char* message, size_t message_length) {
	std::cout << message;
}

static void default_error_logger(const char* message, size_t message_length) {
	std::cerr << message;
}

crop_filter::crop_filter(u32 width, u32 height) :
	crop_filter(width, height, static_cast<logger::callback_t>(default_normal_logger), 
							   static_cast<logger::callback_t>(default_error_logger)) {}

crop_filter::crop_filter(u32 width, u32 height,  
						 logger::callback_t normal_logging_callback, 
						 logger::callback_t error_logging_callback) :
	m_logger(normal_logging_callback, error_logging_callback),
	m_snow_isolator(std::make_unique<isolator>(m_logger, m_stream)),
	m_s3(std::make_unique<blur_detector>(300, 300, m_logger, m_stream)),
	m_width(width),
	m_height(height) {}

crop_filter::~crop_filter() {} // nothing to explicitly destruct for now

void crop_filter::set_normal_logging_callback(logger::callback_t normal_logging_callback) {
	m_logger.m_normal_callback = normal_logging_callback;
}

void crop_filter::set_error_logging_callback(logger::callback_t error_logging_callback) {
	m_logger.m_error_callback = error_logging_callback;
}

void crop_filter::set_blur_threshold(f32 threshold) {
	m_blur_threshold = threshold;
}

void crop_filter::enable_spectral(bool enable) {
	m_spectral_enabled = enable;
}

void crop_filter::enable_spatial(bool enable) {
	m_spatial_enabled = enable;
}

void crop_filter::enable_save_blurry(bool enable) {
	m_blurry_enabled = enable;
}

void crop_filter::enable_save_oversize(bool enable) {
	m_oversize_enabled = enable;
}

u32 crop_filter::crop(const cv::cuda::GpuMat& image) {
	ASSERT_MSG(m_logger, (u32)image.cols == m_width, 
	"crop_filter::crop(): Cannot crop images that are not the same size as what crop_filter was initialized with!");
	ASSERT_MSG(m_logger, (u32)image.rows == m_height, 
	"crop_filter::crop(): Cannot crop images that are not the same size as what crop_filter was initialized with!");
	
	// clear previous images
	m_cropped_images.clear();
	m_cropped_coords.clear();
	m_cropped_sharpness.clear();
	m_blurry_images.clear();
	m_blurry_coords.clear();
	m_blurry_sharpness.clear();
	m_oversize_images.clear();
	m_oversize_coords.clear();
	
	// crop flakes
	u32 number_flakes = m_snow_isolator->isolate_flakes(image);
	if (number_flakes == 0) {
		return 0;
	}
	
	// get crop output
	const std::vector<cv::cuda::GpuMat>& flakes = m_snow_isolator->m_output_flakes;
	const std::vector<std::pair<u32, u32>>& coords = m_snow_isolator->m_output_coords;
	
	// sort into categories
	for (u32 i = 0; i < number_flakes; ++i) {
		// bounds check
		if ((flakes.size() > i) && (coords.size() > i)) {
			// oversize images
			if ((flakes[i].cols > 300) || (flakes[i].rows > 300)) {
				if (m_oversize_enabled) {
					flakes[i].download(m_current_cpu_image, m_stream);
					m_stream.waitForCompletion();
					
					m_oversize_images.emplace_back(m_current_cpu_image.clone());
					m_oversize_coords.emplace_back(coords[i]);
				}
				continue;
			}
			
			// image must be 32-bit float for S3
			flakes[i].convertTo(m_current_image, CV_32F, m_stream);
			f32 sharpness = m_s3->find_S3_max(m_current_image, m_spectral_enabled, m_spatial_enabled);
			
			// resize image to 300x300
			f32 width  = (f32)(300 - flakes[i].cols) / 2.0f;
			f32 height = (f32)(300 - flakes[i].rows) / 2.0f;
			cv::cuda::copyMakeBorder(m_current_image,
									 m_padded_current_image,
									 std::floor(height),
									 std::ceil(height),
									 std::floor(width),
									 std::ceil(width),
									 cv::BORDER_CONSTANT,
									 0,
									 m_stream);
			m_padded_current_image.download(m_current_cpu_image, m_stream);
			m_stream.waitForCompletion();
			
			// original value was 0.575
			if (sharpness > m_blur_threshold) {
				m_cropped_images.emplace_back(m_current_cpu_image.clone());
				m_cropped_coords.emplace_back(coords[i]);
				m_cropped_sharpness.emplace_back(sharpness);
			} else {
				if (m_blurry_enabled) {
					m_blurry_images.emplace_back(m_current_cpu_image.clone());
					m_blurry_coords.emplace_back(coords[i]);
					m_blurry_sharpness.emplace_back(sharpness);
				}
			}
		}
	}
	
	// return number of sharp images
	return m_cropped_images.size();
}

const std::vector<cv::Mat>& crop_filter::get_cropped_images() {
	return m_cropped_images;
}

const std::vector<std::pair<u32, u32>>& crop_filter::get_cropped_coords() {
	return m_cropped_coords;
}

const std::vector<f32>& crop_filter::get_cropped_sharpness() {
	return m_cropped_sharpness;
}

const std::vector<cv::Mat>& crop_filter::get_blurry_images() {
	return m_blurry_images;
}

const std::vector<std::pair<u32, u32>>& crop_filter::get_blurry_coords() {
	return m_blurry_coords;
}

const std::vector<f32>& crop_filter::get_blurry_sharpness() {
	return m_blurry_sharpness;
}

const std::vector<cv::Mat>& crop_filter::get_oversize_images() {
	return m_oversize_images;
}

const std::vector<std::pair<u32, u32>>& crop_filter::get_oversize_coords() {
	return m_oversize_coords;
}
