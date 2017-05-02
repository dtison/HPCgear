cmake_minimum_required(VERSION 3.5)
set(PROJECT_NAME cudaPiEstimator)
project($(PROJECT_NAME))

set(CMAKE_CXX_STANDARD 11)

find_package(CUDA QUIET REQUIRED)


# Options for NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -std=c++11 -Wno-deprecated-gpu-targets
    # -O3?
)

MESSAGE(STATUS "CUDA_NVCC_FLAGS: " ${CUDA_NVCC_FLAGS})

# Include directories  TODO: Check this is it needed?
include_directories(
        ../../include
        $ENV{CUDA_HOME}/samples/common/inc
)


#add_library(gearWorker SHARED ../../../gearWorker/gearWorker.cpp)
#add_library(workerConsoleUI SHARED ../../../workerConsoleUI/workerConsoleUI.cpp)
cuda_add_library(cudaPiEstimator SHARED cudaPiEstimator.cu)




# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
        cuda_worker
        ../../src/gearWorker.cpp
        ##  Are these dependencies: TODO TEST  ../../../inc/gnworker.hpp etc
#        hellocuda.h
#        kernels/hellokernels.cu
#        kernels/hellokernels.h
#        utility/wrapper.cpp
#        utility/wrapper.h
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


#add_executable(
#        cudaPiEstimator_cli
#        cudaPiEstimator.cu
#)

# For linking ...target & libraries to link it with
target_link_libraries(
        cuda_worker
        cudaPiEstimator
 #       gearWorker
        -lboost_program_options
        -lzmq
        -lgearman
        -lcudart
)



# Libraries
#link_libraries(
#        boost_program_options
##        gomp
#        zmq
#        gearman
#        cudart
#)

# Libraries - paths
#link_directories(
#        /opt/foobar/lib
#        /opt/joestuff/lib
#)


#  DO CONSOLE FIRST, THEN RENAME
#set(SOURCE_FILES
#        ../../../workerConsoleUI/workerConsoleUI.cpp)
#set(CLI_NAME $(PROJECT_NAME)_cli)
#add_executable(CLI_NAME ${SOURCE_FILES})

#include_directories(../../../inc)
#add_compile_options()
#add_library()

#add_library(gearWorker OBJECT ../../../gearWorker/gearWorker.cpp)
#add_library(gearWorker STATIC ../../../gearWorker/gearWorker.cpp)