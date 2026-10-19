cmake_minimum_required(VERSION 3.16)

project(awnn_inference LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(AWNN_INFERENCE_BUILD_EXAMPLES "Build awnn_inference example executables" ON)

set(AWNN_PUBLIC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AWNN_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(AWNN_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/vendor)

set(AWNN_VENDOR_SOURCES
    ${AWNN_VENDOR_DIR}/src/npulib.cpp
    ${AWNN_VENDOR_DIR}/src/npu_util.cpp
)

set(AWNN_VENDOR_INCLUDE_DIRS
    ${AWNN_VENDOR_DIR}/include
)

file(GLOB AWNN_VENDOR_RUNTIME_LIBS "${AWNN_VENDOR_DIR}/lib/*.so*")

if(NOT AWNN_VENDOR_RUNTIME_LIBS)
    message(FATAL_ERROR "No bundled vendor runtime libraries found under ${AWNN_VENDOR_DIR}/lib")
endif()

add_library(awnn_inference SHARED
    ${AWNN_SRC_DIR}/session.cpp
    ${AWNN_VENDOR_SOURCES}
)

target_include_directories(awnn_inference
    PUBLIC
        ${AWNN_PUBLIC_INCLUDE_DIR}
    PRIVATE
        ${AWNN_VENDOR_INCLUDE_DIRS}
)

target_link_libraries(awnn_inference PUBLIC ${AWNN_VENDOR_RUNTIME_LIBS} m dl)

if(AWNN_INFERENCE_BUILD_EXAMPLES)
    add_executable(awnn_inference_smoketest
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/smoketest.cpp
    )
    target_link_libraries(awnn_inference_smoketest PRIVATE awnn_inference)
endif()

set_target_properties(awnn_inference PROPERTIES
    BUILD_RPATH "${AWNN_VENDOR_DIR}/lib"
    INSTALL_RPATH "\$ORIGIN"
)

if(AWNN_INFERENCE_BUILD_EXAMPLES)
    set_target_properties(awnn_inference_smoketest PROPERTIES
        BUILD_RPATH "${AWNN_VENDOR_DIR}/lib"
        INSTALL_RPATH "\$ORIGIN/lib"
    )
endif()

include(GNUInstallDirs)

install(TARGETS awnn_inference
    RUNTIME DESTINATION .
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(AWNN_INFERENCE_BUILD_EXAMPLES)
    install(TARGETS awnn_inference_smoketest
        RUNTIME DESTINATION .
    )
endif()

install(DIRECTORY include/ DESTINATION include)
install(FILES ${AWNN_VENDOR_RUNTIME_LIBS} DESTINATION lib)
