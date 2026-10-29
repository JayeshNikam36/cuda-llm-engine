# benchmarks are added here
add_executable(bandwidth bandwidth.cu)
find_package(CUDAToolkit REQUIRED)
add_executable(bench_forward bench_forward.cu)
target_link_libraries(bench_forward PRIVATE engine_core)

add_executable(bench_breakdown bench_breakdown.cu)
target_link_libraries(bench_breakdown PRIVATE engine_core)

add_executable(bench_capacity bench_capacity.cu)
target_link_libraries(bench_capacity PRIVATE engine_core)
