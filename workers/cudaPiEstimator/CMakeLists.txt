cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME cudaPiEstimator)
project($(PROJECT_NAME))

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA QUIET REQUIRED)

#  nvcc Compile Options
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11 -Wno-deprecated-gpu-targets
    # -O3?
)

MESSAGE(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})

# Include directories
include_directories(
        ../../include
        $ENV{CUDA_HOME}/samples/common/inc
 #       /usr/local/cuda/samples/common/inc
)

cuda_add_library(cudaPiEstimator SHARED cudaPiEstimator.cu)

# Target & source files to compile it from
cuda_add_executable(
        cuda_worker
        ../../src/gearWorker.cpp
)

add_executable(
        cuda_worker_cli
        ../../src/workerConsoleUI.cpp
)

target_link_libraries(
        cuda_worker_cli
        cudaPiEstimator
        -lboost_program_options
        -lzmq
        -lgearman
        -lcudart
)


# For linking ...target & libraries to link it with
target_link_libraries(
        cuda_worker
        cudaPiEstimator
        -lboost_program_options
        -lzmq
        -lgearman
        -lcudart
)
