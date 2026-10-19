cmake_minimum_required(VERSION 3.16)

project(awnn_vision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(AWNN_VISION_BUILD_EXAMPLES "Build awnn_vision example executables" ON)

find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV4 REQUIRED opencv4)

set(AWNN_VISION_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(AWNN_VISION_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)

add_library(awnn_vision SHARED
    ${AWNN_VISION_SRC_DIR}/image_preprocessor.cpp
)

target_include_directories(awnn_vision
    PUBLIC
        ${AWNN_VISION_INCLUDE_DIR}
        ${OPENCV4_INCLUDE_DIRS}
)

target_link_directories(awnn_vision PRIVATE ${OPENCV4_LIBRARY_DIRS})
target_compile_options(awnn_vision PRIVATE ${OPENCV4_CFLAGS_OTHER})
target_link_options(awnn_vision PRIVATE ${OPENCV4_LDFLAGS_OTHER})
target_link_libraries(awnn_vision PUBLIC ${OPENCV4_LIBRARIES})

if(AWNN_VISION_BUILD_EXAMPLES)
    add_executable(awnn_vision_smoketest
        ${CMAKE_CURRENT_SOURCE_DIR}/examples/smoketest.cpp
    )
    target_link_libraries(awnn_vision_smoketest PRIVATE awnn_vision)
endif()

include(GNUInstallDirs)

install(TARGETS awnn_vision
    RUNTIME DESTINATION .
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

if(AWNN_VISION_BUILD_EXAMPLES)
    install(TARGETS awnn_vision_smoketest
        RUNTIME DESTINATION .
    )
endif()

install(DIRECTORY include/ DESTINATION include)
