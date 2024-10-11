#pragma once

#ifndef _CUDA_FUNCS_
#define _CUDA_FUNCS_

#include <cstdint>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

void cuda_spatial_map(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream);

void cuda_luminance_map(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream);

void cuda_polar_average(const cv::cuda::GpuMat& input_magnitudes, cv::cuda::GpuMat& output_magnitude_sums, cv::cuda::GpuMat& output_frequencies, cv::cuda::Stream& stream);

void cuda_cv_polyfit(const cv::cuda::GpuMat& src_x, const cv::cuda::GpuMat& src_y, cv::cuda::GpuMat& spectral_map, size_t block_x, size_t block_y, cv::cuda::Stream& stream);

#endif
