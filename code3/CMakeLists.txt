cmake_minimum_required(VERSION 3.5)

project(code3 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCL package
find_package(OpenCL REQUIRED)

# Add the executable target
add_executable(code3 main.cpp)

# Link OpenCL library
target_link_libraries(code3 PRIVATE OpenCL::OpenCL)

install(TARGETS code3
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)


