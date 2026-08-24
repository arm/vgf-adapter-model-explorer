# SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.28)
project(vgf_adapter_model_explorer_native LANGUAGES C CXX)

include(FetchContent)
set(Python3_FIND_VIRTUALENV FIRST)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development.Module)

execute_process(
  COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
  OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
)
if(PYBIND11_CMAKE_DIR_RESULT EQUAL 0 AND EXISTS "${PYBIND11_CMAKE_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
  set(pybind11_DIR "${PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake package directory" FORCE)
endif()

set(FLATBUFFERS_TAG "v25.2.10" CACHE STRING "Pinned FlatBuffers tag")
set(VGF_LIBRARY_TAG "v0.10.0" CACHE STRING "Pinned VGF library revision")

set(FLATBUFFERS_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATC ON CACHE BOOL "" FORCE)
set(FLATBUFFERS_BUILD_FLATHASH OFF CACHE BOOL "" FORCE)
set(FLATBUFFERS_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
  flatbuffers
  GIT_REPOSITORY https://github.com/google/flatbuffers.git
  GIT_TAG ${FLATBUFFERS_TAG}
  GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(flatbuffers)

if(TARGET flatbuffers AND NOT TARGET flatbuffers::flatbuffers)
  add_library(flatbuffers::flatbuffers ALIAS flatbuffers)
endif()
if(TARGET flatc AND NOT TARGET flatbuffers::flatc)
  add_executable(flatbuffers::flatc ALIAS flatc)
endif()

FetchContent_GetProperties(flatbuffers)
set(FLATBUFFERS_PATH "${flatbuffers_SOURCE_DIR}" CACHE PATH "FlatBuffers source directory" FORCE)

set(ML_SDK_VGF_LIB_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(ML_SDK_VGF_LIB_BUILD_PYLIB ON CACHE BOOL "" FORCE)
set(ML_SDK_VGF_LIB_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(ML_SDK_VGF_LIB_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(ML_SDK_VGF_LIB_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(ML_SDK_VGF_LIB_ENABLE_FUZZER OFF CACHE BOOL "" FORCE)
set(ML_SDK_GENERATE_CPACK OFF CACHE BOOL "" FORCE)
set(PYBIND11_FINDPYTHON ON CACHE BOOL "" FORCE)

FetchContent_Declare(
  VGF_Library
  GIT_REPOSITORY https://github.com/arm/ai-ml-sdk-vgf-library.git
  GIT_TAG ${VGF_LIBRARY_TAG}
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(VGF_Library)

add_custom_target(vgf_adapter_native ALL DEPENDS vgfpy)

install(
  TARGETS vgfpy
  LIBRARY DESTINATION .
  RUNTIME DESTINATION .
)
