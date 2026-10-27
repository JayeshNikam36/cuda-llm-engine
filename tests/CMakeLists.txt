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
