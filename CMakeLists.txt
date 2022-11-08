cmake_minimum_required(VERSION 3.0)

project(yolov5_seg)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Release)


include_directories(${PROJECT_SOURCE_DIR})

# opencv 
find_package(OpenCV)
include_directories(${OpenCV_INCLUDE_DIRS})

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
#include_directories(/usr/local/cuda/include)
#link_directories(/usr/local/cuda/lib64)
# tensorrt
#include_directories(/usr/include/arrch64-linux-gnu/)
#link_directories(/usr/lib/arrch64-linux-gnu/)

# 

find_package(CUDA REQUIRED)
message(STATUS " libraries: ${CUDA_LIBRARIES}")
message(STATUS " include path: ${CUDA_INCLUDE_DIRS}")
include_directories(${CUDA_INCLUDE_DIRS})
enable_language(CUDA) 


add_executable(onnx2trt ${PROJECT_SOURCE_DIR}/main1_onnx2trt.cpp)
target_link_libraries(onnx2trt nvinfer)
target_link_libraries(onnx2trt nvonnxparser)
target_link_libraries(onnx2trt cudart)
target_link_libraries(onnx2trt ${OpenCV_LIBS})

add_executable(trt_infer ${PROJECT_SOURCE_DIR}/main2_trt_infer.cpp)
target_link_libraries(trt_infer nvinfer)
target_link_libraries(trt_infer nvonnxparser)
target_link_libraries(trt_infer cudart)
target_link_libraries(trt_infer nvinfer_plugin)
target_link_libraries(trt_infer ${OpenCV_LIBS})


add_definitions(-O2 -pthread)

