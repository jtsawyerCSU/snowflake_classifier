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
#include <opencv2/core/utils/logger.hpp>
#include <opencv2/photo.hpp>
#include "util/savepng.h"
#include <memory>
#include "crop_filter/crop_filter.h"

// shorten filesystem namespace
namespace fs = std::experimental::filesystem;

static void saveImage(const cv::Mat& img, 
					  const std::string& folder, 
					  const std::string& stem, 
					  int num, 
					  const std::pair<int, int>& c, 
					  float sharpness);

int main(int argc, char** argv) {
	
	// opencv likes to print a lot of warnings sometimes
	cv::utils::logging::setLogLevel(cv::utils::logging::LogLevel::LOG_LEVEL_SILENT);
	
	if (argc != 3) {
		std::cout << "Unable to parse arguments!\n";
		std::cout << "Usage: " << argv[0] << " [input folder] [output folder]\n";
		std::exit(-1);
	}
	
	std::string input_folder = argv[1];
	std::string output_folder = argv[2];
	
	if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
		std::cout << "No CUDA capable device found! aborting!\n";
		std::exit(-1);
	}
	
	fs::path input_path{input_folder};
	
	if (!fs::is_directory(input_path)) {
		std::cout << "input folder " << input_folder << " not found! aborting!\n";
		std::exit(-1);
	}
	
	fs::create_directory(output_folder);
	
	std::vector<const char*> camera_name_list = {"cam_0",
												 "cam_1",
												 "cam_2",
												 "cam_3",
												 "cam_4"};
	
	for (std::string camera_name : camera_name_list) {
		
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
		
		std::vector<fs::path> paths;
		cv::Mat image;
		std::unique_ptr<crop_filter> crop_filter_instance = nullptr;
		
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
		
		// sort input images so background subtraction works
		std::sort(paths.begin(), paths.end());
		
		// timer to test how long things take
		timer t;
		
		for (const fs::path& filepath : paths) {
			image = cv::imread(filepath.string(), cv::IMREAD_GRAYSCALE);
			
			// crude SMAS noise cutoff
			// cv::threshold(image, image, 120, 255, cv::THRESH_TOZERO); // 2-2023 images
			// cv::threshold(image, image, 50, 255, cv::THRESH_TOZERO); // 1-2022 images
			
			// some of the SMAS cameras needs these to block out part of the background
			
			// CAM0_1
			// cv::rectangle(foreground, {0, 700}, {450, 1450}, {0, 255, 255}, cv::FILLED);
			// cv::rectangle(foreground, {300, 500}, {400, 750}, {0, 0, 0}, cv::FILLED);
			
			// CAM1_1
			// cv::rectangle(image, {0, 1500}, {0, 350}, {0, 255, 255}, cv::FILLED);
			
			cv::cuda::GpuMat image_gpu;
			image_gpu.upload(image);
			
			std::cout << "Loading " << filepath << '\n';
			
			// the image can be empty sometimes??
			if (image.empty()) {
				continue;
			}
			
			// create crop_filter instance if it doesn't exist yet
			if (crop_filter_instance == nullptr) {
				crop_filter_instance = std::make_unique<crop_filter>(image_gpu.cols, image_gpu.rows);
				
				// disable spectreal sharpness calculations
				crop_filter_instance->enable_spectral(false);
				// without spectral the threshold needs to be a bit lower
				crop_filter_instance->set_blur_threshold(0.13f);
			}
			
			// reset timer
			t.discardTime();
			
			unsigned int number_sharp_flakes = crop_filter_instance->crop(image_gpu);
			
			// print time for this iteration
			t.lap();
			
			// skip saving anything if there are no sharp flakes
			// if (number_sharp_flakes == 0) {
			// 	continue;
			// }
			
			uint32_t snowflake_number = 0;
			
			// save sharp flake images
			const std::vector<cv::Mat>& sharp_flakes = crop_filter_instance->get_cropped_images();
			const std::vector<std::pair<u32, u32>>& sharp_coords = crop_filter_instance->get_cropped_coords();
			const std::vector<f32>& sharp_sharpness = crop_filter_instance->get_cropped_sharpness();
			for (uint32_t i = 0; i < sharp_flakes.size(); ++i) {
				saveImage(sharp_flakes[i], usable_folder, filepath.stem(), snowflake_number + i, sharp_coords[i], sharp_sharpness[i]);
			}
			snowflake_number += sharp_flakes.size();
			
			// save blurry flake images
			const std::vector<cv::Mat>& blurry_flakes = crop_filter_instance->get_blurry_images();
			const std::vector<std::pair<u32, u32>>& blurry_coords = crop_filter_instance->get_blurry_coords();
			const std::vector<f32>& blurry_sharpness = crop_filter_instance->get_blurry_sharpness();
			for (uint32_t i = 0; i < blurry_flakes.size(); ++i) {
				saveImage(blurry_flakes[i], blurry_folder, filepath.stem(), snowflake_number + i, blurry_coords[i], blurry_sharpness[i]);
			}
			snowflake_number += blurry_flakes.size();
			
			// save oversize flake images
			const std::vector<cv::Mat>& oversize_flakes = crop_filter_instance->get_oversize_images();
			const std::vector<std::pair<u32, u32>>& oversize_coords = crop_filter_instance->get_oversize_coords();
			for (uint32_t i = 0; i < oversize_flakes.size(); ++i) {
				saveImage(oversize_flakes[i], oversize_folder, filepath.stem(), snowflake_number + i, oversize_coords[i], 0);
			}
		}
		
	}
	
	return 0;
}

static void saveImage(const cv::Mat& img, const std::string& folder, const std::string& stem, int num, const std::pair<int, int>& c, float sharpness) {
	int cY = c.first;
	int cX = c.second;
	std::string outPath = folder + '/' + stem + 'X' + std::to_string(cX) + 'Y' + std::to_string(cY) + "_S" + std::to_string(sharpness) + "_" + "cropped" + std::to_string(num) + ".png";
	//std::cout << "Saving to " << outPath << '\n';
	static std::vector<int> compression_params{cv::IMWRITE_PNG_COMPRESSION, 9};
	cv::imwrite(outPath, img, compression_params);
}

