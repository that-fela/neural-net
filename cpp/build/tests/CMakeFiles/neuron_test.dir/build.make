# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
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
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/leo/dev/neural-net/cpp

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leo/dev/neural-net/cpp/build

# Include any dependencies generated for this target.
include tests/CMakeFiles/neuron_test.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include tests/CMakeFiles/neuron_test.dir/compiler_depend.make

# Include the progress variables for this target.
include tests/CMakeFiles/neuron_test.dir/progress.make

# Include the compile flags for this target's objects.
include tests/CMakeFiles/neuron_test.dir/flags.make

tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.o: tests/CMakeFiles/neuron_test.dir/flags.make
tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.o: ../tests/test_neuron.cpp
tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.o: tests/CMakeFiles/neuron_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/dev/neural-net/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.o"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.o -MF CMakeFiles/neuron_test.dir/test_neuron.cpp.o.d -o CMakeFiles/neuron_test.dir/test_neuron.cpp.o -c /home/leo/dev/neural-net/cpp/tests/test_neuron.cpp

tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron_test.dir/test_neuron.cpp.i"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leo/dev/neural-net/cpp/tests/test_neuron.cpp > CMakeFiles/neuron_test.dir/test_neuron.cpp.i

tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron_test.dir/test_neuron.cpp.s"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leo/dev/neural-net/cpp/tests/test_neuron.cpp -o CMakeFiles/neuron_test.dir/test_neuron.cpp.s

tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o: tests/CMakeFiles/neuron_test.dir/flags.make
tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o: ../src/n-net.cpp
tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o: tests/CMakeFiles/neuron_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/dev/neural-net/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o -MF CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o.d -o CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o -c /home/leo/dev/neural-net/cpp/src/n-net.cpp

tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron_test.dir/__/src/n-net.cpp.i"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leo/dev/neural-net/cpp/src/n-net.cpp > CMakeFiles/neuron_test.dir/__/src/n-net.cpp.i

tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron_test.dir/__/src/n-net.cpp.s"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leo/dev/neural-net/cpp/src/n-net.cpp -o CMakeFiles/neuron_test.dir/__/src/n-net.cpp.s

tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o: tests/CMakeFiles/neuron_test.dir/flags.make
tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o: ../src/neuron.cpp
tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o: tests/CMakeFiles/neuron_test.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/dev/neural-net/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building CXX object tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o -MF CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o.d -o CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o -c /home/leo/dev/neural-net/cpp/src/neuron.cpp

tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/neuron_test.dir/__/src/neuron.cpp.i"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/leo/dev/neural-net/cpp/src/neuron.cpp > CMakeFiles/neuron_test.dir/__/src/neuron.cpp.i

tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/neuron_test.dir/__/src/neuron.cpp.s"
	cd /home/leo/dev/neural-net/cpp/build/tests && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/leo/dev/neural-net/cpp/src/neuron.cpp -o CMakeFiles/neuron_test.dir/__/src/neuron.cpp.s

# Object files for target neuron_test
neuron_test_OBJECTS = \
"CMakeFiles/neuron_test.dir/test_neuron.cpp.o" \
"CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o" \
"CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o"

# External object files for target neuron_test
neuron_test_EXTERNAL_OBJECTS =

tests/neuron_test: tests/CMakeFiles/neuron_test.dir/test_neuron.cpp.o
tests/neuron_test: tests/CMakeFiles/neuron_test.dir/__/src/n-net.cpp.o
tests/neuron_test: tests/CMakeFiles/neuron_test.dir/__/src/neuron.cpp.o
tests/neuron_test: tests/CMakeFiles/neuron_test.dir/build.make
tests/neuron_test: /usr/lib/x86_64-linux-gnu/libgtest.a
tests/neuron_test: tests/CMakeFiles/neuron_test.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leo/dev/neural-net/cpp/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking CXX executable neuron_test"
	cd /home/leo/dev/neural-net/cpp/build/tests && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neuron_test.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
tests/CMakeFiles/neuron_test.dir/build: tests/neuron_test
.PHONY : tests/CMakeFiles/neuron_test.dir/build

tests/CMakeFiles/neuron_test.dir/clean:
	cd /home/leo/dev/neural-net/cpp/build/tests && $(CMAKE_COMMAND) -P CMakeFiles/neuron_test.dir/cmake_clean.cmake
.PHONY : tests/CMakeFiles/neuron_test.dir/clean

tests/CMakeFiles/neuron_test.dir/depend:
	cd /home/leo/dev/neural-net/cpp/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leo/dev/neural-net/cpp /home/leo/dev/neural-net/cpp/tests /home/leo/dev/neural-net/cpp/build /home/leo/dev/neural-net/cpp/build/tests /home/leo/dev/neural-net/cpp/build/tests/CMakeFiles/neuron_test.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : tests/CMakeFiles/neuron_test.dir/depend

