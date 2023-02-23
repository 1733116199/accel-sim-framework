all: vector_add measure_max_flops

vector_add:
	nvcc -o vector_add vector_add.cu
	./vector_add

measure_max_flops:
	nvcc -o measure_max_flops measure_max_flops.cu
	./measure_max_flops

clean:
	rm -rf vector_add measure_max_flops