cmake_minimum_required(VERSION 3.25)

# The project name should match the target of our mod
project(NeuroDash)

# Verify Geode SDK environment variable is defined
if (NOT DEFINED ENV{GEODE_SDK})
    message(FATAL_ERROR "Unable to find Geode SDK! Please define GEODE_SDK environment variable to point to Geode")
endif()

# Include Geode SDK subdirectory (defines setup_geode_mod)
add_subdirectory($ENV{GEODE_SDK} ${CMAKE_CURRENT_BINARY_DIR}/geode)

# Declare our mod as a shared library and link its source code
add_library(${PROJECT_NAME} SHARED src/main.cpp)

# Setup the Geode target packaging (links bindings, compiles dependencies, and outputs .geode bundle)
setup_geode_mod(${PROJECT_NAME})
