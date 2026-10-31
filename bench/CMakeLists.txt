# benchmarks are added here
add_executable(bandwidth bandwidth.cu)
find_package(CUDAToolkit REQUIRED)
add_executable(bench_forward bench_forward.cu)
target_link_libraries(bench_forward PRIVATE engine_core)

add_executable(bench_breakdown bench_breakdown.cu)
target_link_libraries(bench_breakdown PRIVATE engine_core)

add_executable(bench_capacity bench_capacity.cu)
target_link_libraries(bench_capacity PRIVATE engine_core)

add_executable(bench_attention_decode bench_attention_decode.cu)
target_link_libraries(bench_attention_decode PRIVATE engine_core)

add_executable(bench_attention_compare bench_attention_compare.cu
  ../kernels/attention_decode_fast.cu ../kernels/attention_decode_paged.cu)
target_include_directories(bench_attention_compare PRIVATE ../kernels ../engine)
