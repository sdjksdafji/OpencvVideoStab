# Realtime Video Stablization
## Setup
OpenCv 4.5.1 compiled with the following flags
```
cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_C_COMPILER=/usr/bin/gcc \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D WITH_TBB=ON \
-D WITH_CUDA=ON \
-D BUILD_opencv_cudacodec=OFF \
-D ENABLE_FAST_MATH=1 \
-D CUDA_FAST_MATH=1 \
-D WITH_CUBLAS=1 \
-D WITH_V4L=ON \
-D WITH_QT=OFF \
-D WITH_OPENGL=ON \
-D WITH_GSTREAMER=ON \
-D OPENCV_GENERATE_PKGCONFIG=ON \
-D OPENCV_PC_FILE_NAME=opencv.pc \
-D OPENCV_ENABLE_NONFREE=ON \
-D OPENCV_PYTHON3_INSTALL_PATH=/home/sdjksdafji/anaconda3/envs/opencv4/lib/python3.7/site-packages \
-D OPENCV_EXTRA_MODULES_PATH=~/Documents/others/opencv_contrib/modules \
-D PYTHON3_EXECUTABLE=/home/sdjksdafji/anaconda3/envs/opencv4/bin/python3 \
-D PYTHON_DEFAULT_EXECUTABLE=/home/sdjksdafji/anaconda3/envs/opencv4/bin/python3 \
-D WITH_CUDNN=ON \
-D OPENCV_DNN_CUDA=ON \
-D CUDA_ARCH_BIN=5.2 \
-D BUILD_EXAMPLES=ON ..
```
## Implementation
### Naive Implementation
under ```./naive```, credit to [Nghia Ho](http://nghiaho.com/?p=2093)
- Not realtime
= Using moving average to smooth
### Kalman Filter Implementation
under ```./kalmanFilter```, credit to [Nghia Ho](http://nghiaho.com/?p=2093)
- Realtime
- Using kalman filter to smooth
### Improved Kalman Filter Implementation
under ```./kalmanfilterImproved```, credit to [Nghia Ho](http://nghiaho.com/?p=2093)
- Using the transformation on the current frame. 
(The original version uses the previous frame, which introduces latency by 1 frame)
- Reset the kalman filter when large movement detected. 
The goal is to improve the performance during desired camera movement.