cmake_minimum_required(VERSION 3.14)
project(triton_anchor_linalg LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Pybind11
find_package(pybind11 REQUIRED)

# MLIR & LLVM
find_package(MLIR REQUIRED CONFIG)
list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)
include(TableGen)
include(AddMLIR)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/csrc/include)

# We assume Triton headers are available or we only use MLIR APIs
# This is a skeletal setup for Phase 2.

file(GLOB_RECURSE SRC_FILES 
    "csrc/lib/*.cpp"
    "csrc/bindings/*.cc"
)

pybind11_add_module(_C ${SRC_FILES})

# Link MLIR Libraries
target_link_libraries(_C PRIVATE
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRRewrite
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRArithDialect
    # Add other required MLIR libs
)

set_target_properties(_C PROPERTIES OUTPUT_NAME "_C")
