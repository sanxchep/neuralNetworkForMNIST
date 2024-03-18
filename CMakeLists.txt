cmake_minimum_required(VERSION 3.14)
project(NeuralNetwork)

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Optimization and OpenMP flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fopenmp")

# Eigen
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# Include directories
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(include)

# Find OpenMP
find_package(OpenMP)

# Add executable
add_executable(NeuralNetwork src/train_nn.cpp)

# Link libraries with keyword signatures
if(OpenMP_CXX_FOUND)
    target_link_libraries(NeuralNetwork PUBLIC OpenMP::OpenMP_CXX)
endif()

# Link Eigen with a keyword signature as well
target_link_libraries(NeuralNetwork PUBLIC Eigen3::Eigen)
