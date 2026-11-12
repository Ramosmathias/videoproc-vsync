cmake_minimum_required(VERSION 3.21)
project(VideoProc LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "Cette version de VideoProc nécessite Windows et le compilateur MSVC.")
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc videoio)
find_package(OpenMP REQUIRED)

add_executable(videoproc src/main.cpp)
target_compile_features(videoproc PRIVATE cxx_std_17)
target_compile_options(videoproc PRIVATE /utf-8 /Zc:twoPhase-)
target_compile_definitions(videoproc PRIVATE NOMINMAX _CRT_SECURE_NO_WARNINGS use_norm_compare)
target_include_directories(videoproc PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(videoproc PRIVATE ${OpenCV_LIBS} OpenMP::OpenMP_CXX)

include(CTest)
if(BUILD_TESTING)
    add_test(
        NAME demo_decalage
        COMMAND "${CMAKE_COMMAND}"
            "-DVIDEOPROC_EXE=$<TARGET_FILE:videoproc>"
            "-DDEMO_DIR=${CMAKE_CURRENT_SOURCE_DIR}/reproduction"
            "-DOUTPUT_DIR=${CMAKE_CURRENT_BINARY_DIR}/verification-demo"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/tests/verify-demo.cmake"
    )
endif()
