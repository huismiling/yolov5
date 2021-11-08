# Copyright (C) [2020-2023] The Cambricon Authors. All Rights Reserved.   

cmake_minimum_required(VERSION 3.5)

project(mm_yolox LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)

set(CMAKE_CXX_FLAGS "-Wall -Werror -g -gstabs+ -O0 -fPIC")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

find_package( OpenCV REQUIRED )
include_directories( ${OpenCV_INCLUDE_DIRS} )

include_directories(
    $ENV{NEUWARE_HOME}/include
    ${PROJECT_SOURCE_DIR}/include)

link_directories($ENV{NEUWARE_HOME}/lib64)
link_directories($ENV{LIBTORCH_HOME}/lib/)
link_directories(${OpenCV_LIBRARY_DIRS})

add_executable(mm_yolov5 mm_yolov5.cpp infer_model.cpp sample_buffer.cpp)
target_link_libraries(mm_yolov5 cnrt magicmind ${OpenCV_LIBS})


