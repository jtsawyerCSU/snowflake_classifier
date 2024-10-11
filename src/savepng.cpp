#include "savepng.h"
#include <opencv2/imgcodecs.hpp>

void savepng::save(const std::string& path, cv::Mat& img) {
	static std::vector<int> compression_params{cv::IMWRITE_PNG_COMPRESSION, 9};
	cv::imwrite(path, img, compression_params);
}