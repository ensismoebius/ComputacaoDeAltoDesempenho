cmake_minimum_required(VERSION 3.12)
project(ParallelIntegration)

# Find MPI package
find_package(MPI REQUIRED COMPONENTS CXX)

# Add the executable
add_executable(parallel_integration main.cpp)

# Include MPI headers
target_include_directories(parallel_integration PRIVATE ${MPI_CXX_INCLUDE_DIRS})

# Link MPI libraries
target_link_libraries(parallel_integration PRIVATE ${MPI_CXX_LIBRARIES})

# Set compile options for C++ and MPI
set_target_properties(parallel_integration PROPERTIES
    CXX_STANDARD 11
    COMPILE_FLAGS "${MPI_CXX_COMPILE_FLAGS}"
    LINK_FLAGS "${MPI_CXX_LINK_FLAGS}"
)
