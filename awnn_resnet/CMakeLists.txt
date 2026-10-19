cmake_minimum_required(VERSION 3.16)

project(awnn_resnet LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

set(AWNN_RESNET_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AWNN_RESNET_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(awnn_resnet SHARED
    ${AWNN_RESNET_SRC_DIR}/decoder.cpp
)

target_include_directories(awnn_resnet
    PUBLIC
        ${AWNN_RESNET_INCLUDE_DIR}
        ${OPENCV4_INCLUDE_DIRS}
)

target_link_directories(awnn_resnet PRIVATE ${OPENCV4_LIBRARY_DIRS})
target_compile_options(awnn_resnet PRIVATE ${OPENCV4_CFLAGS_OTHER})
target_link_options(awnn_resnet PRIVATE ${OPENCV4_LDFLAGS_OTHER})
target_link_libraries(awnn_resnet PUBLIC ${OPENCV4_LIBRARIES})

set(AWNN_INFERENCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../awnn_inference" CACHE PATH "Path to awnn_inference")
set(AWNN_VISION_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../awnn_vision" CACHE PATH "Path to awnn_vision")

set(AWNN_INFERENCE_BUILD_EXAMPLES OFF CACHE BOOL "Build awnn_inference example executables" FORCE)
set(AWNN_VISION_BUILD_EXAMPLES OFF CACHE BOOL "Build awnn_vision example executables" FORCE)

add_subdirectory(${AWNN_INFERENCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/awnn_inference_build)
add_subdirectory(${AWNN_VISION_DIR} ${CMAKE_CURRENT_BINARY_DIR}/awnn_vision_build)

add_executable(awnn_resnet_pipeline_demo
    ${CMAKE_CURRENT_SOURCE_DIR}/examples/pipeline_demo.cpp
)
target_link_libraries(awnn_resnet_pipeline_demo PRIVATE awnn_resnet awnn_inference awnn_vision)

include(GNUInstallDirs)

install(TARGETS awnn_resnet
    RUNTIME DESTINATION .
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(TARGETS awnn_resnet_pipeline_demo
    RUNTIME DESTINATION .
)

install(DIRECTORY include/ DESTINATION include)
