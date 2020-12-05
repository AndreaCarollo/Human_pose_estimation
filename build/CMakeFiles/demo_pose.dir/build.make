# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.15

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
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/andrea/Desktop/tmp2/Human_pose_estimation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/andrea/Desktop/tmp2/Human_pose_estimation/build

# Include any dependencies generated for this target.
include CMakeFiles/demo_pose.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/demo_pose.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/demo_pose.dir/flags.make

CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.o: CMakeFiles/demo_pose.dir/flags.make
CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.o: ../src/main_demo_pose.cpp
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/andrea/Desktop/tmp2/Human_pose_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.o"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -o CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.o -c /home/andrea/Desktop/tmp2/Human_pose_estimation/src/main_demo_pose.cpp

CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/andrea/Desktop/tmp2/Human_pose_estimation/src/main_demo_pose.cpp > CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.i

CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/andrea/Desktop/tmp2/Human_pose_estimation/src/main_demo_pose.cpp -o CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.s

# Object files for target demo_pose
demo_pose_OBJECTS = \
"CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.o"

# External object files for target demo_pose
demo_pose_EXTERNAL_OBJECTS =

demo_pose: CMakeFiles/demo_pose.dir/src/main_demo_pose.cpp.o
demo_pose: CMakeFiles/demo_pose.dir/build.make
demo_pose: libfollowmelib.a
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_dnn.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_gapi.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_highgui.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_ml.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_objdetect.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_photo.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_stitching.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_video.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_videoio.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_imgcodecs.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_calib3d.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_features2d.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_flann.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_imgproc.so.4.3.0
demo_pose: /opt/intel/openvino_2020.3.194/opencv/lib/libopencv_core.so.4.3.0
demo_pose: CMakeFiles/demo_pose.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/andrea/Desktop/tmp2/Human_pose_estimation/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable demo_pose"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/demo_pose.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/demo_pose.dir/build: demo_pose

.PHONY : CMakeFiles/demo_pose.dir/build

CMakeFiles/demo_pose.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/demo_pose.dir/cmake_clean.cmake
.PHONY : CMakeFiles/demo_pose.dir/clean

CMakeFiles/demo_pose.dir/depend:
	cd /home/andrea/Desktop/tmp2/Human_pose_estimation/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/andrea/Desktop/tmp2/Human_pose_estimation /home/andrea/Desktop/tmp2/Human_pose_estimation /home/andrea/Desktop/tmp2/Human_pose_estimation/build /home/andrea/Desktop/tmp2/Human_pose_estimation/build /home/andrea/Desktop/tmp2/Human_pose_estimation/build/CMakeFiles/demo_pose.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/demo_pose.dir/depend
