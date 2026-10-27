# benchmarks are added here
add_executable(bandwidth bandwidth.cu)
find_package(CUDAToolkit REQUIRED)
add_executable(bench_forward bench_forward.cu)
target_link_libraries(bench_forward PRIVATE engine_core)
