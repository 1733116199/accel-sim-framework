#include <cuda_runtime.h>
#include <stdio.h>

#define A 1024
#define B 1024
#define N (B * A)

__global__ void vector_add(float *a, float *b1, float * b2, float *c, int size) {
  int i = blockIdx.x * A + threadIdx.x;
  if (i < size) {
    if((i % 2) == 0)
        c[i] = a[i] + b1[i];
    else
        c[i] = a[i] + b2[i];
  }
}

int main() {
    unsigned int size = N * sizeof(float);
  float *a = (float*) malloc(size);
  float *b1 = (float*) malloc(size);
  float *b2 = (float*) malloc(size);
  float *c = (float*) malloc(size);
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b1[i] = 2 * i;
    b2[i] = 3 * i;
  }
  float *a_device, *b1_device, *b2_device, *c_device;
  cudaMalloc(&a_device, size);
  cudaMalloc(&b1_device, size);
  cudaMalloc(&b2_device, size);
  cudaMalloc(&c_device, size);

  cudaMemcpy(a_device, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b1_device, b1, size, cudaMemcpyHostToDevice);
  cudaMemcpy(b2_device, b2, size, cudaMemcpyHostToDevice);

  vector_add<<<B, A>>>(a_device, b1_device, b2_device,c_device, N);
  cudaDeviceSynchronize();
  cudaMemcpy(c, c_device, size, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3 * A; i++) {
    printf("%.2f\n", c[i]);
  }
}