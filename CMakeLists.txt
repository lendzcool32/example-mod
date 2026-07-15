cmake_minimum_required(VERSION 3.25)

project(NeuroDash)

# Locate Geode modding SDK
find_package(Geode REQUIRED)

# Setup Geode mod target
setup_geode_mod(NeuroDash)

# Add source files
target_sources(NeuroDash PRIVATE
    src/main.cpp
)
