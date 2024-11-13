cmake_minimum_required(VERSION 3.16)
project(opuspy)

set(CMAKE_CXX_STANDARD 14)

# Platform-specific configurations
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    # macOS support for both Intel and Apple Silicon
    list(APPEND CMAKE_PREFIX_PATH /opt/homebrew /usr/local)
    set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")  # Support both Intel and M1/M2
elseif (${CMAKE_SYSTEM_NAME} MATCHES "Linux")
    list(APPEND CMAKE_PREFIX_PATH /usr/lib/x86_64-linux-gnu /usr/local)
endif()

# Optimization flags
set(CMAKE_CXX_FLAGS " -Wall -O2")

# Pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

# Find opus libraries and headers
find_library(opus_LIBRARIES opus REQUIRED)
find_path(opus_INCLUDE_DIRS NAMES opus.h PATH_SUFFIXES opus REQUIRED)
include_directories(${opus_INCLUDE_DIRS})

# Find opusfile libraries and headers
find_library(opusfile_LIBRARIES opusfile REQUIRED)
find_path(opusfile_INCLUDE_DIRS NAMES opusfile.h PATH_SUFFIXES opus REQUIRED)
include_directories(${opusfile_INCLUDE_DIRS})

# Find opusenc libraries and headers
find_library(opusenc_LIBRARIES opusenc REQUIRED)
find_path(opusenc_INCLUDE_DIRS NAMES opusenc.h PATH_SUFFIXES opus REQUIRED)
include_directories(${opusenc_INCLUDE_DIRS})

# Find libogg libraries and headers
find_library(ogg_LIBRARIES ogg REQUIRED)
find_path(ogg_INCLUDE_DIRS NAMES ogg/ogg.h PATH_SUFFIXES include REQUIRED)
include_directories(${ogg_INCLUDE_DIRS})

# Debugging messages
message("Opus library: ${opus_LIBRARIES}, include dir: ${opus_INCLUDE_DIRS}")
message("OpusFile library: ${opusfile_LIBRARIES}, include dir: ${opusfile_INCLUDE_DIRS}")
message("OpusEnc library: ${opusenc_LIBRARIES}, include dir: ${opusenc_INCLUDE_DIRS}")
message("Ogg library: ${ogg_LIBRARIES}, include dir: ${ogg_INCLUDE_DIRS}")

# Pybind11 module
pybind11_add_module(opuspy opuspy.cc)
target_link_libraries(opuspy PRIVATE ${opusfile_LIBRARIES} ${opusenc_LIBRARIES} ${opus_LIBRARIES} ${ogg_LIBRARIES})
