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
CMAKE_SOURCE_DIR = /home/leo/dev/neural-net/c

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/leo/dev/neural-net/c/build

# Include any dependencies generated for this target.
include CMakeFiles/neural-net-c.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/neural-net-c.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/neural-net-c.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/neural-net-c.dir/flags.make

CMakeFiles/neural-net-c.dir/main.c.o: CMakeFiles/neural-net-c.dir/flags.make
CMakeFiles/neural-net-c.dir/main.c.o: ../main.c
CMakeFiles/neural-net-c.dir/main.c.o: CMakeFiles/neural-net-c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/dev/neural-net/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building C object CMakeFiles/neural-net-c.dir/main.c.o"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/neural-net-c.dir/main.c.o -MF CMakeFiles/neural-net-c.dir/main.c.o.d -o CMakeFiles/neural-net-c.dir/main.c.o -c /home/leo/dev/neural-net/c/main.c

CMakeFiles/neural-net-c.dir/main.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/neural-net-c.dir/main.c.i"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/leo/dev/neural-net/c/main.c > CMakeFiles/neural-net-c.dir/main.c.i

CMakeFiles/neural-net-c.dir/main.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/neural-net-c.dir/main.c.s"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/leo/dev/neural-net/c/main.c -o CMakeFiles/neural-net-c.dir/main.c.s

CMakeFiles/neural-net-c.dir/src/net.c.o: CMakeFiles/neural-net-c.dir/flags.make
CMakeFiles/neural-net-c.dir/src/net.c.o: ../src/net.c
CMakeFiles/neural-net-c.dir/src/net.c.o: CMakeFiles/neural-net-c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/dev/neural-net/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building C object CMakeFiles/neural-net-c.dir/src/net.c.o"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/neural-net-c.dir/src/net.c.o -MF CMakeFiles/neural-net-c.dir/src/net.c.o.d -o CMakeFiles/neural-net-c.dir/src/net.c.o -c /home/leo/dev/neural-net/c/src/net.c

CMakeFiles/neural-net-c.dir/src/net.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/neural-net-c.dir/src/net.c.i"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/leo/dev/neural-net/c/src/net.c > CMakeFiles/neural-net-c.dir/src/net.c.i

CMakeFiles/neural-net-c.dir/src/net.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/neural-net-c.dir/src/net.c.s"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/leo/dev/neural-net/c/src/net.c -o CMakeFiles/neural-net-c.dir/src/net.c.s

CMakeFiles/neural-net-c.dir/src/neuron.c.o: CMakeFiles/neural-net-c.dir/flags.make
CMakeFiles/neural-net-c.dir/src/neuron.c.o: ../src/neuron.c
CMakeFiles/neural-net-c.dir/src/neuron.c.o: CMakeFiles/neural-net-c.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/leo/dev/neural-net/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Building C object CMakeFiles/neural-net-c.dir/src/neuron.c.o"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -MD -MT CMakeFiles/neural-net-c.dir/src/neuron.c.o -MF CMakeFiles/neural-net-c.dir/src/neuron.c.o.d -o CMakeFiles/neural-net-c.dir/src/neuron.c.o -c /home/leo/dev/neural-net/c/src/neuron.c

CMakeFiles/neural-net-c.dir/src/neuron.c.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing C source to CMakeFiles/neural-net-c.dir/src/neuron.c.i"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -E /home/leo/dev/neural-net/c/src/neuron.c > CMakeFiles/neural-net-c.dir/src/neuron.c.i

CMakeFiles/neural-net-c.dir/src/neuron.c.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling C source to assembly CMakeFiles/neural-net-c.dir/src/neuron.c.s"
	/usr/bin/gcc-12 $(C_DEFINES) $(C_INCLUDES) $(C_FLAGS) -S /home/leo/dev/neural-net/c/src/neuron.c -o CMakeFiles/neural-net-c.dir/src/neuron.c.s

# Object files for target neural-net-c
neural__net__c_OBJECTS = \
"CMakeFiles/neural-net-c.dir/main.c.o" \
"CMakeFiles/neural-net-c.dir/src/net.c.o" \
"CMakeFiles/neural-net-c.dir/src/neuron.c.o"

# External object files for target neural-net-c
neural__net__c_EXTERNAL_OBJECTS =

neural-net-c: CMakeFiles/neural-net-c.dir/main.c.o
neural-net-c: CMakeFiles/neural-net-c.dir/src/net.c.o
neural-net-c: CMakeFiles/neural-net-c.dir/src/neuron.c.o
neural-net-c: CMakeFiles/neural-net-c.dir/build.make
neural-net-c: CMakeFiles/neural-net-c.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/leo/dev/neural-net/c/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_4) "Linking C executable neural-net-c"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/neural-net-c.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/neural-net-c.dir/build: neural-net-c
.PHONY : CMakeFiles/neural-net-c.dir/build

CMakeFiles/neural-net-c.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/neural-net-c.dir/cmake_clean.cmake
.PHONY : CMakeFiles/neural-net-c.dir/clean

CMakeFiles/neural-net-c.dir/depend:
	cd /home/leo/dev/neural-net/c/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/leo/dev/neural-net/c /home/leo/dev/neural-net/c /home/leo/dev/neural-net/c/build /home/leo/dev/neural-net/c/build /home/leo/dev/neural-net/c/build/CMakeFiles/neural-net-c.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/neural-net-c.dir/depend
