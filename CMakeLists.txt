cmake_minimum_required(VERSION 3.17)
Project(matvec)

find_package(OpenCL)
find_package(Threads)

add_executable(${PROJECT_NAME} src/main.cpp)

target_include_directories(${PROJECT_NAME} PUBLIC src)
target_sources(${PROJECT_NAME} PRIVATE
    src/util.h
    src/matrix.h
    src/main.cpp
)
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL Threads::Threads)
