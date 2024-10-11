//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion" // opencv uses enum conversion a lot. the warnings are distracting

#include "S3.h"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda.hpp>
//#include <opencv2/cudaarithm.hpp>
//#include <opencv2/cudaimgproc.hpp>
//#include <opencv2/cudafilters.hpp>
#include <algorithm>
#include <opencv2/cudawarping.hpp>
#include <array>
#include <cmath>
#include <numbers>
#include "cuda_functions.h"
#include "savepng.h"

static cv::cuda::GpuMat spectral_map(const cv::cuda::GpuMat& image, cv::cuda::Stream& stream);
static cv::cuda::GpuMat spatial_map(const cv::cuda::GpuMat& img, cv::cuda::Stream& stream);
static void block_amplitude_spectrum_slope(const cv::cuda::GpuMat& block, cv::cuda::GpuMat& spectral_map, size_t block_x, size_t block_y, cv::cuda::Stream& stream);

#define DEBUG

#ifdef DEBUG
	static int imgnum = 0;
	void save_numberless(const std::string& name, const cv::Mat& img, float scale = 1.f) {
		cv::Mat image_out = img.clone();
		image_out.convertTo(image_out, CV_8U, scale);
		savepng::save("./debug_pics/" + name + ".png", image_out);
	}
	void save(const std::string& name, const cv::Mat& img, float scale = 1.f) {
		save_numberless(std::to_string(imgnum) + "_" + name, img, scale);
	}
#endif

/*
* Input: img is a gray scale image, in float type, range from 0 - 255. 
* You have to convert to gray scale if your image
* is color. You also have to cast img to float in order to run this code
* Output:
* sharpness_estimate: The highest value found in the s3 map
*/
float S3::s3_max(const cv::cuda::GpuMat& image) {
	cv::cuda::Stream stream;

	// ensure the image dimensions are a multiple of 32
	size_t div_x = (image.cols / 32);
	size_t div_y = (image.rows / 32);
	size_t padding_x = image.cols - (div_x * 32);
	size_t padding_y = image.rows - (div_y * 32);
	cv::cuda::GpuMat padded_image;
	cv::cuda::copyMakeBorder(image, padded_image, 0, 0, padding_x, padding_y, cv::BORDER_REFLECT, 0, stream);
	
	// blr_map1
	cv::cuda::GpuMat s_map1  = spectral_map(padded_image, stream);
	// blr_map2
	cv::cuda::GpuMat s_map2 = spatial_map(padded_image, stream);
	
#ifdef DEBUG
	cv::Mat debug_img;
	s_map1.download(debug_img, stream);
	stream.waitForCompletion();
	save("s1", debug_img, 255.f);
	s_map2.download(debug_img, stream);
	stream.waitForCompletion();
	save("s2", debug_img, 255.f);
#endif
	
	float alpha = 0.5f;
	cv::cuda::pow(s_map1,       alpha, s_map1, stream);
	cv::cuda::pow(s_map2, 1.f - alpha, s_map2, stream);
	cv::cuda::GpuMat s3;
	cv::cuda::multiply(s_map1, s_map2, s3, 1, CV_32F, stream);
	
#ifdef DEBUG
	s3.download(debug_img, stream);
	stream.waitForCompletion();
	save("s3", debug_img, 255.f);
	imgnum++;
#endif
	
	stream.waitForCompletion();

	// find max value
	double sharpness_estimate;
	cv::cuda::minMax(s3, nullptr, &sharpness_estimate);
	return sharpness_estimate;
}

/*
* Spectral Sharpness, slope of power spectrum
*/
cv::cuda::GpuMat spectral_map(const cv::cuda::GpuMat& image, cv::cuda::Stream& stream) {
	constexpr uint32_t block_size = 32; // big block size for more coefficients of the power spectrum
	constexpr uint32_t delta_block = block_size / 4; // Distance b/w blocks
	constexpr uint32_t offset = delta_block / 2;
	constexpr uint32_t border_width = block_size / 2;

	cv::cuda::GpuMat padded_image;
	cv::cuda::copyMakeBorder(image, padded_image, border_width, border_width, border_width, border_width, cv::BORDER_REFLECT, 0, stream);

	uint32_t num_cols = image.size().width  / delta_block;
	uint32_t num_rows = image.size().height / delta_block;
	
	cv::Size map_size(num_cols, num_rows);
	cv::cuda::GpuMat luminance(map_size, CV_32F);
	cuda_luminance_map(padded_image, luminance, stream);
	
	cv::Rect sbb{}; // sub block bounds
	sbb.x = 0;
	sbb.y = 0;
	sbb.width = block_size;
	sbb.height = block_size;
	
	cv::cuda::GpuMat spectrum(map_size, CV_32F);
	
	// 32 concurrent kernels is the maximum in CUDA
	// static std::vector<cv::cuda::Stream> block_streams(32);
	
	//block_amplitude_spectrum_slope(padded_image(sbb), spectrum, 0, 0, stream);
	
	// size_t next_block_stream = 0;
	for (uint32_t y = 0; y < num_rows; ++y) {
		for (uint32_t x = 0; x < num_cols; ++x) {
			sbb.x = x * delta_block + offset;
			sbb.y = y * delta_block + offset;
			
			block_amplitude_spectrum_slope(padded_image(sbb), spectrum, x, y, stream);
			// block_amplitude_spectrum_slope(padded_image(sbb), spectrum, x, y, block_streams[next_block_stream]);
			// if (++next_block_stream >= block_streams.size()) {
			// 	next_block_stream = 0;
			// }
		}
	}
	
	// for (cv::cuda::Stream& block_stream : block_streams) {
	// 	block_stream.waitForCompletion();
	// }
	//stream.waitForCompletion();
	
	cv::cuda::multiply(spectrum, luminance, spectrum, 1, -1, stream);
	
	cv::cuda::GpuMat temp;
	cv::cuda::resize(spectrum, temp, {}, delta_block, delta_block, cv::INTER_NEAREST, stream);
	return temp;
}

/*
* Spatial Sharpness, local total variation
*/
cv::cuda::GpuMat spatial_map(const cv::cuda::GpuMat& image, cv::cuda::Stream& stream) {
	constexpr uint32_t block_size = 8;
	constexpr uint32_t offset = block_size / 2;

	cv::Size map1_size(image.size().width / block_size, image.size().height / block_size);
	cv::cuda::GpuMat spatial_map1(map1_size, CV_32F);
	cuda_spatial_map(image, spatial_map1, stream);

	cv::cuda::GpuMat intermidiate_map1;
	cv::cuda::resize(spatial_map1, intermidiate_map1, {}, 2, 2, cv::INTER_NEAREST, stream);

	cv::cuda::GpuMat padded_image;
	cv::cuda::copyMakeBorder(image, padded_image, offset, offset, offset, offset, cv::BORDER_REFLECT, 0, stream);
	cv::Size map2_size(padded_image.size().width / block_size, padded_image.size().height / block_size);
	cv::cuda::GpuMat spatial_map2(map2_size, CV_32F);
	cuda_spatial_map(padded_image, spatial_map2, stream);

	cv::cuda::GpuMat intermidiate_map2;
	cv::cuda::resize(spatial_map2, intermidiate_map2, {}, 2, 2, cv::INTER_NEAREST, stream);

	cv::Rect map2_adjust_box(cv::Point(1, 1), intermidiate_map1.size());
	cv::cuda::max(intermidiate_map2(map2_adjust_box), intermidiate_map1, intermidiate_map1, stream);

	cv::cuda::GpuMat final_map;
	cv::cuda::resize(intermidiate_map1, final_map, {}, block_size / 2, block_size / 2, cv::INTER_NEAREST, stream);

	return final_map;
}

// least squares approximation algorithm
static void cvPolyfit(cv::Mat& src_x, cv::Mat& src_y, cv::Mat& dst, int order) {
	cv::Mat M = cv::Mat::zeros(src_x.rows, order + 1, CV_32FC1);
	cv::Mat copy;
	for (int i = 0; i <= order; i++) {
		copy = src_x.clone();
		cv::pow(copy, i, copy);
		cv::Mat M1 = M.col(i);
		copy.col(0).copyTo(M1);
	}
	cv::Mat M_t;
	cv::transpose(M, M_t);
	cv::Mat M_t_M = M_t * M;
	cv::Mat M_t_M_inv;
	cv::invert(M_t_M, M_t_M_inv);
	cv::Mat r = M_t_M_inv * M_t * src_y;
	dst = r.clone();
}

void block_amplitude_spectrum_slope(const cv::cuda::GpuMat& block, cv::cuda::GpuMat& spectral_map, size_t block_x, size_t block_y, cv::cuda::Stream& stream) {
	static constexpr size_t block_size = 32;
	static cv::cuda::GpuMat hanning_window;
	static cv::Ptr<cv::cuda::DFT> cuda_dft_instance;
	if (hanning_window.empty()) {
		cv::Mat window;
		cv::createHanningWindow(window, cv::Size(block_size, block_size), CV_32F);
		cv::pow(window, 1.75f, window);
		hanning_window.upload(window);

		cuda_dft_instance = cv::cuda::createDFT(cv::Size(block_size, block_size), 0);
	}
	cv::cuda::GpuMat block_window_product;
	cv::cuda::multiply(block, hanning_window, block_window_product, 1, CV_32F, stream);

	cv::cuda::GpuMat combined_complex;
	cuda_dft_instance->compute(block_window_product, combined_complex, stream);
	cv::cuda::GpuMat split_complex[2];
	cv::cuda::split(combined_complex, split_complex, stream);

	cv::cuda::GpuMat magnitudes_gpu;
	cv::cuda::magnitude(split_complex[0], split_complex[1], magnitudes_gpu, stream);

	cv::cuda::log(magnitudes_gpu, magnitudes_gpu, stream);

	cv::cuda::GpuMat frequencies_gpu{};
	cv::cuda::GpuMat magnitude_sums_gpu{};

	cuda_polar_average(magnitudes_gpu, magnitude_sums_gpu, frequencies_gpu, stream);

	uint32_t as_rows = (block_size / 2) - 1;
	cv::cuda::divide(magnitude_sums_gpu, cv::Scalar(as_rows), magnitude_sums_gpu, 1, -1, stream);
	
	cv::cuda::log(frequencies_gpu, frequencies_gpu, stream);
	cv::cuda::log(magnitude_sums_gpu, magnitude_sums_gpu, stream);
	
	cuda_cv_polyfit(frequencies_gpu, magnitude_sums_gpu, spectral_map, block_x, block_y, stream);
}
