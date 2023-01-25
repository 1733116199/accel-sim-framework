#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

__global__ void vector_add(float *a, float *b, float *c, int size) {
  int i = threadIdx.x;
  if (i < size) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  float a[N];
  float b[N];
  float c[N];
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b[i] = 2 * i;
  }
  float *a_device, *b_device, *c_device;
  cudaMalloc(&a_device, sizeof(a));
  cudaMalloc(&b_device, sizeof(b));
  cudaMalloc(&c_device, sizeof(c));

  cudaMemcpy(a_device, a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpy(b_device, b, sizeof(b), cudaMemcpyHostToDevice);

  vector_add<<<1, N>>>(a_device, b_device, c_device, N);
  cudaMemcpy(c, c_device, sizeof(c), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    printf("%.2f\n", c[i]);
  }
}