cmake_minimum_required(VERSION 2.8.0)
project(demo)

set(CMAKE_BUILD_TYPE Debug)

# OPENCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# OPENSSL
find_package(OpenSSL REQUIRED)  
include_directories(${OPENSSL_INCLUDE_DIR}) 

find_package(CUDA REQUIRED)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")  
    add_compile_options(-Wno-deprecated-declarations)  
endif() 
CUDA_ADD_EXECUTABLE(demo main.cpp kernel.cu)

target_link_libraries(demo ${OpenCV_LIBS} ${OPENSSL_LIBRARIES})