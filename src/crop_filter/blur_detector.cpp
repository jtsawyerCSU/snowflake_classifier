//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion" // opencv uses enum conversion a lot. the warnings are distracting

#include "crop_filter/blur_detector.h"
#include <iostream>
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
#include <array>
#include <cmath>
#include <numbers>
#include "crop_filter/cuda/cuda_functions.h"
#include "util/savepng.h"
#include "crop_filter/cuda/cuda_polyfit.h"
#include "crop_filter/core/asserts.h"

//#define DEBUG

#ifdef DEBUG

static int imgnum = 0;
static void save_numberless(const std::string& name, const cv::Mat& img, float scale = 1.f) {
	cv::Mat image_out = img.clone();
	image_out.convertTo(image_out, CV_8U, scale);
	savepng::save("./debug_pics/" + name + ".png", image_out);
}

static void save(const std::string& name, const cv::Mat& img, float scale = 1.f) {
	save_numberless(std::to_string(imgnum) + "_" + name, img, scale);
}

#endif

blur_detector::blur_detector(u32 width, u32 height, logger& logger, cv::cuda::Stream& stream) : 
	m_logger(logger),
	m_stream(stream) {
	
	// reallocate/resize GpuMats
	resize(width, height);
	
	// setup members
	// generate hanning window
	constexpr u32 spectral_block_size = 32;
	cv::Mat window;
	cv::createHanningWindow(window, cv::Size(spectral_block_size, spectral_block_size), CV_32F);
	cv::pow(window, 1.75f, window);
	m_hanning_window.upload(window);
	
	// create cuda dft instance
	m_cuda_dft = cv::cuda::createDFT(cv::Size(spectral_block_size, spectral_block_size), 0);
}

blur_detector::~blur_detector() {
	// nothing needs manually destructed for now
}

void blur_detector::resize(u32 width, u32 height) {
	m_image_width = width;
	m_image_height = height;
	// pad image size to a multiple of 32
	u32 div_x = (m_image_width / 32);
	u32 div_y = (m_image_height / 32);
	m_padding_x = m_image_width - (div_x * 32);
	m_padding_y = m_image_height - (div_y * 32);
	
	m_map_size = cv::Size((m_image_width + m_padding_x), (m_image_height + m_padding_y));
	m_padded_input_image = cv::cuda::GpuMat(m_map_size, CV_32F);
	m_spectral_map = cv::cuda::GpuMat(m_map_size, CV_32F);
	m_spatial_map = cv::cuda::GpuMat(m_map_size, CV_32F);
	
	// preallocate GpuMats for spectral map calculations
	constexpr u32 spectral_block_size = 32;
	constexpr u32 spectral_delta_block = spectral_block_size / 4;
	
	m_padded_spectral_image_size = cv::Size((m_map_size.width + spectral_block_size), (m_map_size.height + spectral_block_size));
	m_padded_spectral_image = cv::cuda::GpuMat(m_padded_spectral_image_size, CV_32F);
	
	m_internal_spectral_map_size = cv::Size(m_map_size.width / spectral_delta_block, m_map_size.height / spectral_delta_block);
	m_spectral_luminance_map = cv::cuda::GpuMat(m_internal_spectral_map_size, CV_32F);
	m_spectral_slope_map = cv::cuda::GpuMat(m_internal_spectral_map_size, CV_32F);
	
	kernel = cv::Rect(0, 0, spectral_block_size, spectral_block_size);
	
	// preallocate GpuMats for spatial map calculations
	constexpr u32 spatial_block_size = 8;

	m_partial_spatial_map1_size = cv::Size(m_map_size.width / spatial_block_size, m_map_size.height / spatial_block_size);
	m_partial_spatial_map1 = cv::cuda::GpuMat(m_partial_spatial_map1_size, CV_32F);
	
	m_intermidiate_map1_size = cv::Size(m_partial_spatial_map1_size.width * 2, m_partial_spatial_map1_size.height * 2);
	m_intermidiate_map1 = cv::cuda::GpuMat(m_intermidiate_map1_size, CV_32F);
	
	m_padded_spatial_image_size = cv::Size((m_map_size.width + spatial_block_size), (m_map_size.height + spatial_block_size));
	m_padded_spatial_image = cv::cuda::GpuMat(m_padded_spatial_image_size, CV_32F);
	
	m_partial_spatial_map2_size = cv::Size(m_padded_spatial_image_size.width / spatial_block_size, m_padded_spatial_image_size.height / spatial_block_size);
	m_partial_spatial_map2 = cv::cuda::GpuMat(m_partial_spatial_map2_size, CV_32F);
	
	m_intermidiate_map2_size = cv::Size(m_partial_spatial_map2_size.width * 2, m_partial_spatial_map2_size.height * 2);
	m_intermidiate_map2 = cv::cuda::GpuMat(m_intermidiate_map2_size, CV_32F);

	m_intermidiate_map2_adjust_box = cv::Rect(cv::Point(1, 1), m_intermidiate_map1.size());
}

/*
* Input: image is a gray scale image, in float type, range from 0 - 255. 
* You have to convert to gray scale if your image
* is color. You also have to cast img to float in order to run this code
* Output:
* sharpness_estimate: The highest value found in the s3 map
*/
f32 blur_detector::find_S3_max(const cv::cuda::GpuMat& image, bool enable_spectral, bool enable_spatial) {
	// check that one option is enabled
	if (!enable_spectral && !enable_spatial) {
		// if nothing is enabled, just return maximum sharpness value
		return 1.f;
	}
	
	// resize image to preallocated size
	f32 width  = (f32)(m_image_width + m_padding_x - image.cols) / 2.0f;
	f32 height = (f32)(m_image_height + m_padding_y - image.rows) / 2.0f;
	cv::cuda::copyMakeBorder(image, 
							 m_padded_input_image, 
							 std::floor(height), 
							 std::ceil(height), 
							 std::floor(width), 
							 std::ceil(width), 
							 cv::BORDER_CONSTANT, 
							 0, 
							 m_stream);
	
	if (enable_spectral) {
		generate_spectral_map();
	}
	if (enable_spatial) {
		generate_spatial_map();
	}
	
#ifdef DEBUG
	cv::Mat debug_img;
	if (enable_spectral) {
		m_spectral_map.download(debug_img, m_stream);
		m_stream.waitForCompletion();
		save("s1", debug_img, 255.f);
	}
	if (enable_spatial) {
		m_spatial_map.download(debug_img, m_stream);
		m_stream.waitForCompletion();
		save("s2", debug_img, 255.f);
	}
#endif
	
	if (enable_spectral && enable_spatial) {
		f32 alpha = 0.5f;
		cv::cuda::pow(m_spectral_map, alpha, m_spectral_map, m_stream);
		cv::cuda::pow(m_spatial_map, 1.f - alpha, m_spatial_map, m_stream);
		cv::cuda::multiply(m_spectral_map, m_spatial_map, m_s3_map, 1, CV_32F, m_stream);
	}
	
#ifdef DEBUG
	if (enable_spectral && enable_spatial) {
		m_s3_map.download(debug_img, m_stream);
		m_stream.waitForCompletion();
		save("s3", debug_img, 255.f);
	}
	imgnum++;
#endif
	
	m_stream.waitForCompletion();
	
	// find max value
	f64 sharpness_estimate = 0.f;
	
	if (enable_spectral && enable_spatial) {
		cv::cuda::minMax(m_s3_map, nullptr, &sharpness_estimate);
	} else {
		if (enable_spectral) {
			cv::cuda::minMax(m_spectral_map, nullptr, &sharpness_estimate);
		}
		if (enable_spatial) {
			cv::cuda::minMax(m_spatial_map, nullptr, &sharpness_estimate);
		}
	}
	
	return sharpness_estimate;
}

/*
* Spectral Sharpness, slope of power spectrum
*/
void blur_detector::generate_spectral_map() {
	constexpr uint32_t block_size = 32;
	constexpr uint32_t delta_block = block_size / 4; // Distance b/w blocks
	constexpr uint32_t offset = delta_block / 2;
	constexpr uint32_t border_width = block_size / 2;

	cv::cuda::copyMakeBorder(m_padded_input_image, m_padded_spectral_image, border_width, border_width, border_width, border_width, cv::BORDER_REFLECT, 0, m_stream);

	uint32_t num_cols = m_padded_input_image.size().width  / delta_block;
	uint32_t num_rows = m_padded_input_image.size().height / delta_block;
	
	cuda_luminance_map(m_padded_spectral_image, m_spectral_luminance_map, m_stream);
	
	for (uint32_t y = 0; y < num_rows; ++y) {
		for (uint32_t x = 0; x < num_cols; ++x) {
			kernel.x = x * delta_block + offset;
			kernel.y = y * delta_block + offset;
			
			amplitude_spectrum_slope(m_padded_spectral_image(kernel), cuda_cv_mat_ptr(m_spectral_slope_map, x, y));
		}
	}
	
	cv::cuda::multiply(m_spectral_slope_map, m_spectral_luminance_map, m_spectral_slope_map, 1, -1, m_stream);
	
	cv::cuda::resize(m_spectral_slope_map, m_spectral_map, {}, delta_block, delta_block, cv::INTER_NEAREST, m_stream);
}

/*
* Spatial Sharpness, local total variation
*/
void blur_detector::generate_spatial_map() {
	constexpr u32 block_size = 8;
	constexpr u32 offset = block_size / 2;
	
	cuda_spatial_map(m_padded_input_image, m_partial_spatial_map1, m_stream);
	
	cv::cuda::resize(m_partial_spatial_map1, m_intermidiate_map1, {}, 2, 2, cv::INTER_NEAREST, m_stream);
	
	cv::cuda::copyMakeBorder(m_padded_input_image, m_padded_spatial_image, offset, offset, offset, offset, cv::BORDER_REFLECT, 0, m_stream);
	
	cuda_spatial_map(m_padded_spatial_image, m_partial_spatial_map2, m_stream);
	
	cv::cuda::resize(m_partial_spatial_map2, m_intermidiate_map2, {}, 2, 2, cv::INTER_NEAREST, m_stream);
	
	cv::cuda::max(m_intermidiate_map2(m_intermidiate_map2_adjust_box), m_intermidiate_map1, m_intermidiate_map1, m_stream);
	
	cv::cuda::resize(m_intermidiate_map1, m_spatial_map, {}, block_size / 2, block_size / 2, cv::INTER_NEAREST, m_stream);
}

void blur_detector::amplitude_spectrum_slope(const cv::cuda::GpuMat& block, f32* spectral_map_ptr) {
	cv::cuda::multiply(block, m_hanning_window, m_block_window_product, 1, CV_32F, m_stream);
	
	m_cuda_dft->compute(m_block_window_product, m_combined_complex, m_stream);
	cv::cuda::split(m_combined_complex, m_split_complex, m_stream);
	
	cv::cuda::magnitude(m_split_complex[0], m_split_complex[1], m_magnitudes_gpu, m_stream);
	
	cv::cuda::log(m_magnitudes_gpu, m_magnitudes_gpu, m_stream);
	
	u32 n = std::min(m_magnitudes_gpu.rows, m_magnitudes_gpu.cols);
	u32 radii = (n / 2) - 1;
	if (((u32)m_magnitude_sums_gpu.rows != radii) || (m_magnitude_sums_gpu.cols != 1)) {
		m_magnitude_sums_gpu = cv::cuda::GpuMat(radii, 1, CV_32F);
	}
	if (((u32)m_frequencies_gpu.rows != radii) || (m_frequencies_gpu.cols != 1)) {
		m_frequencies_gpu = cv::cuda::GpuMat(radii, 1, CV_32F);
	}
	
	cuda_polar_average(m_magnitudes_gpu, m_magnitude_sums_gpu, m_frequencies_gpu, m_stream);
	
	constexpr u32 block_size = 32;
	u32 amplitude_spectrum_rows = (block_size / 2) - 1;
	cv::cuda::divide(m_magnitude_sums_gpu, cv::Scalar(amplitude_spectrum_rows), m_magnitude_sums_gpu, 1, -1, m_stream);
	
	cv::cuda::log(m_frequencies_gpu, m_frequencies_gpu, m_stream);
	cv::cuda::log(m_magnitude_sums_gpu, m_magnitude_sums_gpu, m_stream);
	
	if ((m_gpu_solver == nullptr) || (m_gpu_solver->m_points != (u32)m_frequencies_gpu.rows)) {
		// create cuda polynomial solver instance
		m_gpu_solver = std::make_unique<cuda_polyfit>(m_frequencies_gpu.rows, m_logger);
		m_gpu_solver->set_stream(m_stream);
	}
	
	m_gpu_solver->solve(m_frequencies_gpu, m_magnitude_sums_gpu, spectral_map_ptr);
}
