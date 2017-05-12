# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.5

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:


#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:


# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list


# Suppress display of executed commands.
$(VERBOSE).SILENT:


# A target that is always out of date.
cmake_force:

.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/me/devel/HPCgear/workers/cudaPiEstimator

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/me/devel/HPCgear/workers/cudaPiEstimator

# Include any dependencies generated for this target.
include CMakeFiles/cuda_worker.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/cuda_worker.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/cuda_worker.dir/flags.make

CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o: CMakeFiles/cuda_worker.dir/flags.make
CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o: /home/me/devel/HPCgear/src/gearWorker.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/me/devel/HPCgear/workers/cudaPiEstimator/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o -c /home/me/devel/HPCgear/src/gearWorker.cpp

CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/me/devel/HPCgear/src/gearWorker.cpp > CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.i

CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/me/devel/HPCgear/src/gearWorker.cpp -o CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.s

CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.requires:

.PHONY : CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.requires

CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.provides: CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.requires
	$(MAKE) -f CMakeFiles/cuda_worker.dir/build.make CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.provides.build
.PHONY : CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.provides

CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.provides.build: CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o


# Object files for target cuda_worker
cuda_worker_OBJECTS = \
"CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o"

# External object files for target cuda_worker
cuda_worker_EXTERNAL_OBJECTS =

cuda_worker: CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o
cuda_worker: CMakeFiles/cuda_worker.dir/build.make
cuda_worker: /usr/local/cuda/lib64/libcudart_static.a
cuda_worker: /usr/lib/x86_64-linux-gnu/librt.so
cuda_worker: libcudaPiEstimator.so
cuda_worker: /usr/local/cuda/lib64/libcudart_static.a
cuda_worker: /usr/lib/x86_64-linux-gnu/librt.so
cuda_worker: CMakeFiles/cuda_worker.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/me/devel/HPCgear/workers/cudaPiEstimator/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable cuda_worker"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/cuda_worker.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/cuda_worker.dir/build: cuda_worker

.PHONY : CMakeFiles/cuda_worker.dir/build

CMakeFiles/cuda_worker.dir/requires: CMakeFiles/cuda_worker.dir/home/me/devel/HPCgear/src/gearWorker.cpp.o.requires

.PHONY : CMakeFiles/cuda_worker.dir/requires

CMakeFiles/cuda_worker.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/cuda_worker.dir/cmake_clean.cmake
.PHONY : CMakeFiles/cuda_worker.dir/clean

CMakeFiles/cuda_worker.dir/depend:
	cd /home/me/devel/HPCgear/workers/cudaPiEstimator && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/me/devel/HPCgear/workers/cudaPiEstimator /home/me/devel/HPCgear/workers/cudaPiEstimator /home/me/devel/HPCgear/workers/cudaPiEstimator /home/me/devel/HPCgear/workers/cudaPiEstimator /home/me/devel/HPCgear/workers/cudaPiEstimator/CMakeFiles/cuda_worker.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/cuda_worker.dir/depend

