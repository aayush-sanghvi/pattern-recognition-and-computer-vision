cmake_minimum_required(VERSION 3.15.0)
project(project-4 VERSION 0.1.0 LANGUAGES C CXX)

include(CTest)
enable_testing()

add_executable(project-4 augmentedReality.cpp calibration.cpp)
find_package(OpenCV REQUIRED PATHS "/usr/include/opencv4/opencv2/")
include_directories($(OpenCV_INCLUDE_DIRS))

target_include_directories(project-4 PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(project-4 ${OpenCV_LIBS})

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)