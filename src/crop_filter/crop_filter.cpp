#include "crop_filter/crop_filter.h"
#include "crop_filter/core/asserts.h"
#include <cstdio>

static void default_logger(const char* message, size_t message_length) {
	std::printf(message);
}

crop_filter::crop_filter(u32 width, u32 height) :
	m_width(width),
	m_height(height),
	m_logger(default_logger),
	m_s3(std::make_unique<blur_detector>(width, height)),
	m_snow_isolator(std::make_unique<blur_detector>()) {}

crop_filter::crop_filter(u32 width, u32 height, std::function<void (const char*)> logging_callback) :
	crop_filter(width, height) {
	set_logging_callback(logging_callback);
}

crop_filter::crop_filter(u32 width, u32 height, std::function<void (const char*, size_t)> logging_callback) :
	crop_filter(width, height) {
	set_logging_callback(logging_callback);
}

crop_filter::~crop_filter() {} // nothing to explicitly destruct for now

void crop_filter::set_logging_callback(std::function<void (const char*)> logging_callback) {
	set_logging_callback(
		[=](const char* message, size_t message_length) -> void {
			logging_callback(message);
		});
}

void crop_filter::set_logging_callback(std::function<void (const char*, size_t)> logging_callback) {
	m_logger = logger(logging_callback);
}

void crop_filter::set_blur_threshold(f32 threshold) {
	m_blur_threshold = threshold;
}

u32 crop_filter::crop(const cv::cuda::GpuMat& image) {
	ASSERT_MSG(m_logger, image.cols != m_width, 
	"crop_filter::crop(): Cannot crop images that are not the same size as what crop_filter was initialized with!");
	ASSERT_MSG(m_logger, image.rows != m_height, 
	"crop_filter::crop(): Cannot crop images that are not the same size as what crop_filter was initialized with!");
	
	u32 number_flakes = snowisolator.isolateFlakes(image);
	if (number_flakes == 0) {
		return 0;
	}
	
	
	cv::Mat flake;
	cv::cuda::GpuMat flake_gpu;
	cv::cuda::Stream stream;
	for (uint32_t i = 0; i < flakesperimage; ++i) {
		if ((flakes.size() > i) && (coords.size() > i)) {
			if ((flakes[i].cols > 300) || (flakes[i].rows > 300)) {
				flakes[i].download(flake, stream);
				stream.waitForCompletion();
				saveImage(flake, oversize_folder, filepath.stem(), i, coords[i], 0);
				continue;
			}
			
			float width  = (float)(300 - flakes[i].cols) / 2.0f;
			float height = (float)(300 - flakes[i].rows) / 2.0f;
			cv::cuda::copyMakeBorder(flakes[i],
									flake_gpu,
									std::floor(height),
									std::ceil(height),
									std::floor(width),
									std::ceil(width),
									cv::BORDER_CONSTANT,
									0,
									stream);
			
			if ((S3 == nullptr) || ((S3->m_image_width != (u32)flake_gpu.cols) || (S3->m_image_height != (u32)flake_gpu.rows))) {
				S3 = std::make_unique<blur_detector>(flake_gpu.cols, flake_gpu.rows);
			}
			if ((S3->m_image_width != (u32)flake_gpu.cols) || (S3->m_image_height != (u32)flake_gpu.rows)) {
				S3->resize(flake_gpu.cols, flake_gpu.rows);
			}
			
			// image must be 32-bit float for S3
			flake_gpu.convertTo(flake_gpu, CV_32F);
			float sharpness = S3->find_S3_max(flake_gpu);
			
			flake_gpu.download(flake, stream);
			stream.waitForCompletion();
			
			// original value was 0.575
			if (sharpness > 0.2) {
				saveImage(flake, usable_folder, filepath.stem(), i, coords[i], sharpness);
			} else {
				saveImage(flake, blurry_folder, filepath.stem(), i, coords[i], sharpness);
			}
		}
	}
	
	flakes.clear();
	coords.clear();
	
	
	
}

std::span<cv::cuda::GpuMat> crop_filter::get_cropped_images() {
	return m_cropped_images;
}

std::span<cv::cuda::GpuMat> crop_filter::get_cropped_coords() {
	return m_cropped_coords;
}

std::span<cv::cuda::GpuMat> crop_filter::get_blurry_images() {
	return m_blurry_images;
}

std::span<cv::cuda::GpuMat> crop_filter::get_blurry_coords() {
	return m_blurry_coords;
}
