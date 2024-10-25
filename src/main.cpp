//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion" // opencv uses enum conversion a lot. the warnings are distracting

#include <iostream>
#include <experimental/filesystem>
#include <cstdint>
#include <vector>
#include <string>
#include <algorithm>
#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include "util/timer.h"
#include <sstream>
#include <fstream>
#include "crop_filter/blur_detector.h"
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/photo.hpp>
#include "util/savepng.h"
#include "crop_filter/Isolator.h"
#include <memory>

namespace fs = std::experimental::filesystem;

//#define DEBUG

#ifdef DEBUG
	static std::string debugfolder = "./debug_pics";
#endif

static void saveImage(cv::Mat& img, const std::string& folder, const std::string& stem, int num, const std::pair<int, int>& c, float sharpness);

int main(int argc, char** argv) {

	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	
	if (argc != 3) {
		std::cout << "Unable to parse arguments!\n";
		std::cout << "Usage: " << argv[0] << " [input folder] [output folder]\n";
		std::exit(-1);
	}
	
	// std::string input_folder = "./pics";
	std::string input_folder = argv[1];
	std::string output_folder = argv[2];
	//std::string camera_name = argv[3];

	if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
		std::cout << "No CUDA capable device found! aborting!\n";
		std::exit(-1);
	}
	
	fs::path input_path{input_folder};
	
	if (!fs::is_directory(input_path)) {
		std::cout << input_folder << " folder not found! aborting!\n";
		std::exit(-1);
	}
	
	fs::create_directory(output_folder);
	
#ifdef DEBUG
	fs::create_directory(debugfolder);
#endif
	
	std::vector<const char*> camera_name_list = {"Cam0_1",
												 "Cam1_1",
												 "Cam1_2",
												 "Cam2_1",
												 "Cam3_1",
												 "Cam4_1",
												 "Cam5_1",
												 "Cam6_1"};
	
	for (const char* camera_nam : camera_name_list) {
		std::string camera_name = camera_nam;
		
		
		std::string usable_folder   = output_folder + "/usable_images";
		std::string unusable_folder = output_folder + "/unusable_images";
		std::string oversize_folder = output_folder + "/unusable_images/oversize_images";
		std::string blurry_folder   = output_folder + "/unusable_images/blurry_images";
		
		fs::create_directory(unusable_folder);
		fs::create_directory(oversize_folder);
		fs::create_directory(blurry_folder);
		fs::create_directory(usable_folder);
		
		oversize_folder	+= "/" + camera_name;
		blurry_folder	+= "/" + camera_name;
		usable_folder	+= "/" + camera_name;
		
		fs::create_directory(oversize_folder);
		fs::create_directory(blurry_folder);
		fs::create_directory(usable_folder);
		
		cv::Mat image;
		std::vector<std::pair<int, int>> coords;
		std::vector<cv::cuda::GpuMat> flakes;
		uint32_t flakesperimage = 0;
		Isolator snowisolator;
		std::vector<fs::path> paths;
		std::unique_ptr<blur_detector> S3;
		
		for (const fs::directory_entry& entry : fs::recursive_directory_iterator{input_path}) {
			fs::path filepath = entry.path();
			
			if (fs::is_directory(filepath)) {
				continue;
			}
			
			if (filepath.string().find(camera_name) == std::string::npos) {
				continue;
			}
			
			paths.emplace_back(filepath);
		}
		
		std::sort(paths.begin(), paths.end());
		
		// Timer t;
		// double flakes_per_average = 0;
		// size_t images_count = 0;

		for (const fs::path& filepath : paths) {
			image = cv::imread(filepath.string(), cv::IMREAD_GRAYSCALE);
			
			// // one of the SMAS cameras needs this to block out part of the background
			// cv::threshold(image, image, 120, 255, cv::THRESH_TOZERO); // 2-2023 images
			cv::threshold(image, image, 50, 255, cv::THRESH_TOZERO); // 1-2022 images

			// cv::rectangle(image, {0, 1500}, {0, 350}, {0, 255, 255}, cv::FILLED);
			
			cv::cuda::GpuMat image_gpu;
			image_gpu.upload(image);
			image_gpu.convertTo(image_gpu, CV_32F);

			std::cout << "Loading " << filepath << '\n';

			// std::cout << "\n\n\n\n";
			
			// if ((S3 == nullptr) || ((S3->m_image_width != (u32)image_gpu.cols) || (S3->m_image_height != (u32)image_gpu.rows))) {
			// 	S3 = std::make_unique<blur_detector>(image_gpu.cols, image_gpu.rows);
			// }
			
			// t.discardTime();
			// float sharpness = S3->find_S3_max(image_gpu);
			// t.lap();

			// the image can be empty sometimes??
			if (image.empty()) {
				continue;
			}
			image.convertTo(image, CV_8U);

			if (snowisolator.needsbackground) {
				snowisolator.setBackground(image);
				continue;
			}

			bool is_cam0_1 = (filepath.string().find("CAM0_1") != std::string::npos);

			snowisolator.isolateFlakes(image, flakes, flakesperimage, coords, is_cam0_1);

			// flakes_per_average *= images_count;
			// images_count += 1;
			// flakes_per_average += flakesperimage;
			// flakes_per_average /= images_count;

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
			flakesperimage = 0;
		}
		
	}
	
	// std::cout << "flakes_per_average: " << flakes_per_average << '\n';

	return 0;
}

static void saveImage(cv::Mat& img, const std::string& folder, const std::string& stem, int num, const std::pair<int, int>& c, float sharpness) {
	int cY = c.first;
	int cX = c.second;
	std::string outPath = folder + '/' + stem + 'X' + std::to_string(cX) + 'Y' + std::to_string(cY) + "_S" + std::to_string(sharpness) + "_" + "cropped" + std::to_string(num) + ".png";
	//std::cout << "Saving to " << outPath << '\n';
	static std::vector<int> compression_params{cv::IMWRITE_PNG_COMPRESSION, 9};
	cv::imwrite(outPath, img, compression_params);
}

