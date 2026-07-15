
cmake_minimum_required(VERSION 3.25)

project(NeuroDash)

# Verify Geode SDK environment variable is defined
if (NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "Unable to find Geode SDK! Please define GEODE_SDK environment variable to point to Geode")
endif()

# Include Geode SDK subdirectory (this defines setup_geode_mod and all SDK logic)
add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode)

# Setup Geode mod target
setup_geode_mod(NeuroDash)

# Add source files
target_sources(NeuroDash PRIVATE
    src/main.cpp
)
