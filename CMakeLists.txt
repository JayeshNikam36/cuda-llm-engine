cmake_minimum_required(VERSION 3.22)

# Must come before project(): project() fills in a default architecture otherwise,
# and a parent project (NeuroServe) must be able to choose its own.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 89)
endif()

project(cuda_engine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

# ---- The engine as one library ------------------------------------------
add_library(cuda_engine STATIC
    engine/layer.cu
    engine/model.cu
    engine/cached.cu
    engine/paged.cu
    engine/batched.cu
    engine/scheduler.cu
    kernels/embedding.cu
    kernels/rmsnorm.cu
    kernels/rope.cu
    kernels/swiglu.cu
    kernels/residual_add.cu
    kernels/linear.cu
    kernels/attention_naive.cu
    kernels/attention_decode_naive.cu
    kernels/attention_decode_paged.cu
    kernels/attention_decode_fast.cu
    kernels/kv_scatter.cu
    kernels/argmax.cu)
add_library(cuda_engine::core ALIAS cuda_engine)

target_include_directories(cuda_engine PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/engine
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels)

target_link_libraries(cuda_engine PUBLIC CUDA::cublas CUDA::cudart)

# Position-independent code so it can be linked into a Python extension module later.
set_target_properties(cuda_engine PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ---- Tests and benchmarks only when this is the top-level project ---------
if(PROJECT_IS_TOP_LEVEL)
    enable_testing()
    add_subdirectory(tests)
    add_subdirectory(bench)
endif()