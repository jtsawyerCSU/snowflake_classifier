//#pragma GCC diagnostic ignored "-Wdeprecated-enum-enum-conversion" // opencv uses enum conversion a lot. the warnings are distracting

#include "Isolator.h"
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
#include "savepng.h"

constexpr double scalar = 8;
constexpr double invscalar = 1 / scalar;
constexpr int pointExpand = 5;
constexpr int boxExpand = 20;

void Isolator::setBackground(const cv::Mat& background) {
	bgs[lastbg].upload(background);
	cv::cuda::rshift(bgs[lastbg], cv::Scalar(2), bgs[lastbg]);
	if (++lastbg >= 4) {
		--lastbg;
		cv::cuda::add(bgs[0], bgs[1], bg);
		cv::cuda::add(bgs[2], bg,     bg);
		cv::cuda::add(bgs[3], bg,     bg);
		needsbackground = false;
	}
}

void Isolator::isolateFlakes(const cv::Mat& input, std::vector<cv::cuda::GpuMat>& output, uint32_t& flakes, std::vector<std::pair<int, int>>& outputcoords, const bool isCAM0_1) {
	
	if (needsbackground) {
		std::cerr << "Cannot isolate snowflakes without first supplying a background image!\n";
	}
	
	const cv::Mat& foreground = input;
	
	if (isCAM0_1) {
		cv::rectangle(foreground, {0, 700}, {450, 1450}, {0, 255, 255}, cv::FILLED);
		cv::rectangle(foreground, {300, 500}, {400, 750}, {0, 0, 0}, cv::FILLED);
	}
	
	fg.upload(foreground, stream);
	
	cv::cuda::subtract(fg, bg, img2, {}, {}, stream);
	
	cv::cuda::resize(img2, img1, {}, invscalar, invscalar, cv::INTER_LINEAR, stream);
	
	cv::cuda::threshold(img1, img, 30, 255, cv::THRESH_BINARY, stream);
	
	img.download(img_cpu, stream);
	
	// can put a thread sleep here with completion interrupt
	stream.waitForCompletion();
	
	cv::cuda::bitwise_or(fg, fg, img1, img2, stream);
	
	cv::findNonZero(img_cpu, points);
	
	stream.waitForCompletion();
	
	//std::vector<cv::cuda::GpuMat> flakeImgs;
	
	//getBoundedImgs(img1, flakeImgs, 15, outputcoords);
	getBoundedImgs(img1, output, 15, outputcoords);
	
	// here
	
	//getCrop(flakeImgs, output);
	//flakes = flakeImgs.size();
	flakes = output.size();
	
	//std::swap(bg, fg);
	points.clear();
	
	// add new image to background
	
	if (++lastbg >= 4) {
		lastbg = 0;
	}
	
	cv::cuda::subtract(bg, bgs[lastbg], bg);
	std::swap(bgs[lastbg], fg);
	cv::cuda::rshift(bgs[lastbg], cv::Scalar(2), bgs[lastbg]);
	cv::cuda::add(bgs[lastbg], bg, bg);
}

void Isolator::getCrop(std::vector<cv::cuda::GpuMat>& flakeImgs, std::vector<cv::Mat>& output) {
	///*
	int initialsize = output.size();
	output.resize(initialsize + flakeImgs.size());
	
	for (unsigned int i = 0; i < flakeImgs.size(); ++i) {
		flakeImgs[i].download(output[initialsize + i], stream);
	}
	stream.waitForCompletion();
	//*/
	/*
	std::vector<cv::cuda::GpuMat> temp(flakeImgs.size());
	int initialsize = output.size();
	output.resize(initialsize + flakeImgs.size());
	for (unsigned int i = 0; i < flakeImgs.size(); ++i) {
		float width  = (float)(300 - flakeImgs[i].cols) / 2.0f;
		float height = (float)(300 - flakeImgs[i].rows) / 2.0f;
		cv::cuda::copyMakeBorder(flakeImgs[i],
					 temp[i],
					 std::floor(height),
					 std::ceil(height),
					 std::floor(width),
					 std::ceil(width),
					 cv::BORDER_CONSTANT,
					 0,
					 stream);
		temp[i].download(output[initialsize + i], stream);
	}
	stream.waitForCompletion();
	*/
}

struct boundingbox {
	boundingbox() = default;
	boundingbox(const boundingbox&) = default;
	boundingbox(cv::Point p) : minX{p.x - pointExpand}, minY{p.y - pointExpand}, maxX{p.x + pointExpand}, maxY{p.y + pointExpand} {}
	cv::Rect operator()() const {
		cv::Rect r{};
		r.x = minX;
		r.y = minY;
		r.width = maxX - minX;
		r.height = maxY - minY;
		return r;
	}
	bool operator==(const boundingbox& other) const {
		return !((maxX < other.minX) || (minX > other.maxX) || (maxY < other.minY) || (minY > other.maxY));
	}
	void combine(const boundingbox& other) {
		minX = std::min(minX, other.minX);
		maxX = std::max(maxX, other.maxX);
		minY = std::min(minY, other.minY);
		maxY = std::max(maxY, other.maxY);
	}
	int minX{}, minY{}, maxX{}, maxY{};
};

void Isolator::getBoundedImgs(cv::cuda::GpuMat& img, std::vector<cv::cuda::GpuMat>& flakeImgs, int threshold, std::vector<std::pair<int, int>>& coords) {
	if (points.empty()) {
		return;
	}
	std::vector<boundingbox> boxes(points.begin(), points.end());
loop:
	for (unsigned int i = 0; i < (boxes.size() - 1); ++i) {
		boundingbox& combinedBox = boxes[i];
		std::vector<boundingbox>::iterator it = std::remove_if(boxes.begin() + i + 1, boxes.end(), 
			[&](const boundingbox& box) {
				if (combinedBox == box) {
					combinedBox.combine(box);
					return true;
				} else {
					return false;
				}
			});
		if (it != boxes.end()) {
			boxes.erase(it, boxes.end());
			goto loop;
		}
	}
	flakeImgs.resize(boxes.size());
	coords.resize(boxes.size());
	int j = 0;
	for (const boundingbox& combinedBox : boxes) {
		cv::Rect boundingRectangle = combinedBox();
		boundingRectangle.x *= scalar;
		boundingRectangle.y *= scalar;
		boundingRectangle.width *= scalar;
		boundingRectangle.height *= scalar;
		if ((boundingRectangle.width < threshold) || (boundingRectangle.height < threshold)) {
			continue;
		}
		boundingRectangle.x -= boxExpand;
		boundingRectangle.y -= boxExpand;
		boundingRectangle.width += boxExpand << 1;
		boundingRectangle.height += boxExpand << 1;
		if (boundingRectangle.x < 0 ||
			boundingRectangle.y < 0 ||
			boundingRectangle.x + boundingRectangle.width > img.cols ||
			boundingRectangle.y + boundingRectangle.height > img.rows) {
			continue;
		}
		int centerX = boundingRectangle.x + boundingRectangle.width / 2;
		int centerY = boundingRectangle.y + boundingRectangle.height / 2;
		coords[j] = {centerX, centerY};
		flakeImgs[j] = img(boundingRectangle);
		++j;
	}
	flakeImgs.resize(j);
	coords.resize(j);
}






















