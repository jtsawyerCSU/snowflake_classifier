#pragma once

#ifndef _cuda_functions_H_
#define _cuda_functions_H_

#include "crop_filter/core/defines.h"
#include <opencv2/core/mat.hpp>
#include <opencv2/core/cuda.hpp>

void cuda_spatial_map(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream);

void cuda_luminance_map(const cv::cuda::GpuMat& input, cv::cuda::GpuMat& output, cv::cuda::Stream& stream);

void cuda_polar_average(const cv::cuda::GpuMat& input_magnitudes, cv::cuda::GpuMat& output_magnitude_sums, cv::cuda::GpuMat& output_frequencies, cv::cuda::Stream& stream);

void cuda_copy_cv_mat(const cv::cuda::GpuMat& cv_mat, f32* device_mem, size_t device_mem_size, void* cuda_stream);

void cuda_copy_spectrum_value(f32* spectrum_ptr, f32* value, void* cuda_stream);

f32* cuda_cv_mat_ptr(const cv::cuda::GpuMat& cv_mat, u32 x, u32 y);

#endif
