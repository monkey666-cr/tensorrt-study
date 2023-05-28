# 最低版本要求
cmake_minimum_required(VERSION 3.10)

# 项目信息
project(trt_demo LANGUAGES CXX CUDA)

# 添加 CMAKE_MODULE_PATH, 否则找不到 FindTensorRT.cmake
# set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")
list (APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# 寻找TensorRT库
find_package(TensorRT REQUIRED)

if(TensorRT_FOUND)
    message(STATUS "Found TensorRT ${TensorRT_VERSION} in ${TensorRT_ROOT_DIR}")
    message(STATUS "TensorRT libraries: ${TensorRT_LIBRARIES}")
    message(STATUS "TensorRT include files: ${TensorRT_INCLUDE_DIR}")
else()
    message(FATAL_ERROR "Cannot find TensorRT")
endif()

# 添加可执行文件
add_executable(build src/build.cpp)

# 头文件
target_include_directories(build PRIVATE ${TensorRT_INCLUDE_DIRS})
# 链接库
target_link_libraries(build PRIVATE ${TensorRT_LIBRARIES})

# 添加可执行文件
add_executable(runtime src/runtime.cu)

# 头文件
target_include_directories(runtime PRIVATE ${TensorRT_INCLUDE_DIRS})
# 链接库
target_link_libraries(runtime PRIVATE ${TensorRT_LIBRARIES})
