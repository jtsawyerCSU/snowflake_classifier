## Snowflake Classifier project

### Requirements:
- CMake
- CUDA 11.8 (11.8 and 11.5 are confirmed working)

### to compile:
```
cd snowflake_classifier/build
```
```
cmake -DCMAKE_CUDA_COMPILER="/usr/local/cuda-11.8/bin/nvcc" -DCUDA_GENERATION=Auto ..
```
```
cmake --build .
```

### example CUDA 11.8 installation:
```
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
```
```
sudo dpkg -i cuda-keyring_1.1-1_all.deb
```
```
sudo apt-get update
```
```
sudo apt-get -y install cuda-toolkit-11-8
```

after manually installing CUDA you will need to set the cmake flag 
```
-DCMAKE_CUDA_COMPILER="/usr/local/cuda-11.8/bin/nvcc"
```

to drastically reduce compile time you can restrict the CUDA archetecture that gets compiled to whatever is detected on your machine with this cmake flag
```
-DCUDA_GENERATION=Auto
```

if getting the compilation error "error: parameter packs not expanded with ‘...’:" you need to update nvcc to 11.8
more information here:
- https://developer.nvidia.com/cuda-11-8-0-download-archive

and here:
- https://stackoverflow.com/questions/74350584/nvcc-compilation-error-using-thrust-in-cuda-11-5

if needed here is a command to completely purge all things nvidia: 
```
sudo apt-get --purge remove "*cublas*" "*cufft*" "*curand*" "*cusolver*" "*cusparse*" "*npp*" "*nvjpeg*" "cuda*" "nsight*"
```
