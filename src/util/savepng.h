#pragma once

#ifndef _savepng_H
#define _savepng_H

#include <string>
#include <opencv2/core/mat.hpp>

namespace savepng {
	
	void save(const std::string& path, cv::Mat& img);
	
};

#endif
