cmake_minimum_required(VERSION 3.16)
Project(matvec)

find_package(OpenCL)
find_package(Threads)
find_package(Catch2 REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp)

add_library(main)

target_include_directories(main PUBLIC src)
target_include_directories(main PUBLIC ext)
target_sources(main PRIVATE
    src/application.h
    src/application.cpp

    src/util.h
    src/util.cpp
    src/matrix.h
)
target_link_libraries(main OpenCL::OpenCL Threads::Threads)
target_link_libraries(${PROJECT_NAME} main)

add_executable(tests test/test_main.cpp)
target_include_directories(tests PUBLIC src)
target_link_libraries(tests PRIVATE Catch2Main Catch2::Catch2 main)
