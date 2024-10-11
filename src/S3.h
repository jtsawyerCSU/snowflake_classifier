#pragma once

#ifndef _S3_H
#define _S3_H

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
#include <cstdint>
#include <tuple>

namespace S3 {
	
	float s3_max(const cv::cuda::GpuMat& img);
	
}

#endif
