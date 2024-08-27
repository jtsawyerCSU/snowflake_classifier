#include <cstdio>
#include <opencv2/core/core.hpp>
#include <opencv2/core/cuda.hpp>

int main() {
    std::printf("hello world\n");

	if (cv::cuda::getCudaEnabledDeviceCount() < 1) {
        std::printf("no CUDA capable device detected!\n");
	} else {
        std::printf("CUDA capable device detected!\n");
        std::printf("OpenCV with CUDA successfully built!\n");
    }


    return 0;
}
