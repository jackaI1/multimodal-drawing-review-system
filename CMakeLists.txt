cmake_minimum_required(VERSION 3.18)
project(redlinecheck_pixels LANGUAGES NONE)

option(REDLINECHECK_BUILD_CPP "Build the optional C++ pixel comparison" ON)
if(REDLINECHECK_BUILD_CPP)
  enable_language(CXX)
  find_package(Python REQUIRED COMPONENTS Interpreter Development.Module)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_pixels_cpp cpp/pixels.cpp)
  target_compile_features(_pixels_cpp PRIVATE cxx_std_17)
  set_target_properties(_pixels_cpp PROPERTIES CXX_EXTENSIONS OFF)
  install(TARGETS _pixels_cpp LIBRARY DESTINATION redlinecheck
                              RUNTIME DESTINATION redlinecheck)
endif()
