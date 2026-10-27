add_executable(hello hello.cu)
add_test(NAME hello COMMAND hello)

add_executable(test_embedding test_embedding.cu ../kernels/embedding.cu)
target_include_directories(test_embedding PRIVATE ../kernels ../engine)
add_test(NAME embedding COMMAND test_embedding)
