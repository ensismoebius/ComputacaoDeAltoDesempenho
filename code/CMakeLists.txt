cmake_minimum_required(VERSION 3.5)

project(IntegralEvaluator LANGUAGES CXX)

find_package(OpenMP REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(IntegralEvaluator main.cpp)

target_link_libraries(IntegralEvaluator PUBLIC OpenMP::OpenMP_CXX)

install(TARGETS IntegralEvaluator
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
