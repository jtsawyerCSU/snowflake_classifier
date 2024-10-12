#pragma once

#ifndef _Isolator_H
#define _Isolator_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafilters.hpp>
#include <vector>
#include <array>

struct Isolator {
	
	void setBackground(const cv::Mat& background);
	
	void isolateFlakes(const cv::Mat& input, std::vector<cv::cuda::GpuMat>& output, uint32_t& flakes, std::vector<std::pair<int, int>>& outputcoords, const bool isCAM0_1);
	
private:
	
	void getCrop(std::vector<cv::cuda::GpuMat>& flakeImgs, std::vector<cv::Mat>& output);

	void getBoundedImgs(cv::cuda::GpuMat& img, std::vector<cv::cuda::GpuMat>& flakeImgs, int threshold, std::vector<std::pair<int, int>>& coords);
	
	std::vector<cv::Point> points;
	cv::cuda::Stream stream;
	std::array<cv::cuda::GpuMat, 4> bgs;
	cv::cuda::GpuMat bg;
	cv::cuda::GpuMat fg;
	cv::cuda::GpuMat img;
	cv::cuda::GpuMat img1;
	cv::cuda::GpuMat img2;
	cv::Mat img_cpu;
	uint8_t lastbg{};
public:
	bool needsbackground{true};
	
};

#endif
