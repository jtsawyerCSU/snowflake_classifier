cmake_minimum_required(VERSION 2.8.12)
project(snowflake_classifier LANGUAGES CXX CUDA)

# Require C++17, and disable compiler-specific extensions (if possible).
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD 17)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# Set build flags for CXX
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS_DEBUG}")

# Set build flags for CUDA
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall")
set(CMAKE_CUDA_FLAGS_DEBUG "-g")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3")
set(CMAKE_CUDA_FLAGS_RELWITHDEBINFO "${CMAKE_CUDA_FLAGS_RELEASE} ${CMAKE_CUDA_FLAGS_DEBUG}")
set(CMAKE_CUDA_ARCHITECTURES "50;52;60;61;70;75;80;86;89;90")

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${PROJECT_SOURCE_DIR}/.git")
# Update submodules as needed
    option(GIT_SUBMODULE "Check submodules during build" ON)
    if(GIT_SUBMODULE)
        message(STATUS "Submodule update")
        execute_process(COMMAND ${GIT_EXECUTABLE} submodule update --init --recursive
                        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
                        RESULT_VARIABLE GIT_SUBMOD_RESULT)
        if(NOT GIT_SUBMOD_RESULT EQUAL "0")
            message(FATAL_ERROR "git submodule update --init --recursive failed with ${GIT_SUBMOD_RESULT}, please checkout submodules")
        endif()
    endif()
endif()

if(NOT EXISTS "${PROJECT_SOURCE_DIR}/libs/opencv/CMakeLists.txt")
    message(FATAL_ERROR "The submodules were not downloaded! GIT_SUBMODULE was turned off or failed. Please update submodules and try again.")
endif()

# Set up opencv
# see https://docs.opencv.org/4.x/db/d05/tutorial_config_reference.html
string(CONCAT OPENCV_CUDA_MODULES_PATHS "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudaarithm;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudabgsegm;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudacodec;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudafeatures2d;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudafilters;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudaimgproc;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudalegacy;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudaobjdetect;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudaoptflow;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudastereo;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudawarping;"
                                        "${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudev;")
set(OPENCV_EXTRA_MODULES_PATH ${OPENCV_CUDA_MODULES_PATHS} CACHE PATH "" FORCE)
set(BUILD_SHARED_LIBS OFF)
set(BUILD_TESTS OFF)
set(BUILD_PERF_TESTS OFF)
set(BUILD_opencv_apps OFF)
set(WITH_JPEG OFF)
set(WITH_TIFF OFF)
set(WITH_WEBP OFF)
set(WITH_OPENJPEG OFF)
set(WITH_JASPER OFF)
set(WITH_OPENEXR OFF)
set(WITH_FFMPEG OFF)
set(WITH_V4L OFF)
set(WITH_GSTREAMER OFF) # note: this might be needed for image capture later
set(WITH_MSMF OFF)
set(VIDEOIO_ENABLE_PLUGINS OFF)
set(OpenMP ON)
set(PARALLEL_ENABLE_PLUGINS OFF)
set(WITH_GTK OFF)
set(WITH_WIN32UI OFF)
set(HIGHGUI_ENABLE_PLUGINS OFF)
set(BUILD_JAVA OFF)
set(BUILD_opencv_python2 OFF)
set(BUILD_opencv_python3 OFF)
set(WITH_IMGCODEC_HDR OFF)
set(WITH_IMGCODEC_SUNRASTER OFF)
set(WITH_IMGCODEC_PXM OFF)
set(WITH_IMGCODEC_PFM OFF)
set(WITH_NVCUVID OFF)
set(WITH_NVCUVENC OFF)
set(WITH_VTK OFF)
set(WITH_GTK OFF)
set(ENABLE_FAST_MATH ON)
set(BUILD_opencv_world ON)
set(WITH_CUDA ON)
set(WITH_CUFFT ON)
set(HAVE_CUBLAS ON)
set(CUDA_FAST_MATH ON)
set(WITH_CUDNN OFF)
set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Installation Directory" FORCE)
add_subdirectory(libs/opencv EXCLUDE_FROM_ALL)

# Add executables
file(GLOB_RECURSE PROJECT_SOURCE_FILES CONFIGURE_DEPENDS "src/*.cpp" "src/*.cu")
add_executable(snowflake_classifier ${PROJECT_SOURCE_FILES})
target_include_directories(snowflake_classifier PRIVATE ${PROJECT_SOURCE_DIR}/src
                                           ${PROJECT_SOURCE_DIR}/build
                                           ${PROJECT_SOURCE_DIR}/libs/opencv/modules/core/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudev/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv/modules/imgcodecs/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv/modules/imgproc/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudaarithm/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudaimgproc/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudafilters/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv_contrib/modules/cudawarping/include
                                           ${PROJECT_SOURCE_DIR}/libs/opencv/modules/photo/include
                                           /usr/local/cuda/include)
link_directories(snowflake_classifier /usr/local/cuda/lib64)
target_link_libraries(snowflake_classifier PRIVATE opencv_world
                                                   stdc++fs
                                                   cusolver_static
                                                   cudart_static
                                                   culibos
                                                   cusolver_lapack_static
                                                   metis_static
                                                   cublas_static
                                                   cublasLt_static
                                                   cusparse_static)
