rbk : reduce_by_key.cu
	nvcc -std=c++14 -g $^ gtest/libgtest.a -o rbk

clean :
	rm -f rbk