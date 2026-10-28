add_executable(hello hello.cu)
add_test(NAME hello COMMAND hello)

add_executable(test_embedding test_embedding.cu ../kernels/embedding.cu)
target_include_directories(test_embedding PRIVATE ../kernels ../engine)
add_test(NAME embedding COMMAND test_embedding)

add_executable(test_rmsnorm test_rmsnorm.cu ../kernels/rmsnorm.cu)
target_include_directories(test_rmsnorm PRIVATE ../kernels ../engine)
add_test(NAME rmsnorm COMMAND test_rmsnorm)

add_executable(test_rope test_rope.cu ../kernels/rope.cu)
target_include_directories(test_rope PRIVATE ../kernels ../engine)
add_test(NAME rope COMMAND test_rope)

add_executable(test_swiglu test_swiglu.cu ../kernels/swiglu.cu)
target_include_directories(test_swiglu PRIVATE ../kernels ../engine)
add_test(NAME swiglu COMMAND test_swiglu)

find_package(CUDAToolkit REQUIRED)
add_executable(test_linear test_linear.cu ../kernels/linear.cu)
target_include_directories(test_linear PRIVATE ../kernels ../engine)
target_link_libraries(test_linear PRIVATE CUDA::cublas)
add_test(NAME linear COMMAND test_linear)

add_executable(test_attention test_attention.cu ../kernels/attention_naive.cu)
target_include_directories(test_attention PRIVATE ../kernels ../engine)
add_test(NAME attention COMMAND test_attention)

add_executable(test_residual_add test_residual_add.cu ../kernels/residual_add.cu)
target_include_directories(test_residual_add PRIVATE ../kernels ../engine)
add_test(NAME residual_add COMMAND test_residual_add)

add_library(engine_core STATIC
  ../engine/layer.cu ../kernels/rmsnorm.cu ../kernels/linear.cu ../kernels/rope.cu
  ../kernels/attention_naive.cu ../kernels/swiglu.cu ../kernels/residual_add.cu)
target_include_directories(engine_core PUBLIC ../kernels ../engine)
target_link_libraries(engine_core PUBLIC CUDA::cublas)

add_executable(test_layer0 test_layer0.cu)
target_link_libraries(test_layer0 PRIVATE engine_core)
add_test(NAME layer0 COMMAND test_layer0)

target_sources(engine_core PRIVATE ../engine/model.cu)
add_executable(test_generate test_generate.cu)
target_link_libraries(test_generate PRIVATE engine_core)
add_test(NAME generate COMMAND test_generate)

target_sources(engine_core PRIVATE ../kernels/embedding.cu)

target_sources(engine_core PRIVATE ../engine/cached.cu ../kernels/attention_decode_naive.cu)
add_executable(test_generate_cached test_generate_cached.cu)
target_link_libraries(test_generate_cached PRIVATE engine_core)
add_test(NAME generate_cached COMMAND test_generate_cached)

add_executable(test_paged_attention test_paged_attention.cu
  ../kernels/attention_decode_paged.cu ../kernels/attention_decode_naive.cu)
target_include_directories(test_paged_attention PRIVATE ../kernels ../engine)
add_test(NAME paged_attention COMMAND test_paged_attention)

