#include <cuda_runtime.h>
#include <stdio.h>

#define N 1024

__global__ void vector_add(float *a, float *b1, float * b2, float *c, int size) {
  int i = threadIdx.x;
  if (i < size) {
    if((i % 2) == 0)
        c[i] = a[i] + b1[i];
    else
        c[i] = a[i] + b2[i];
  }
}

int main() {
  float a[N];
  float b1[N];
  float b2[N];
  float c[N];
  for (int i = 0; i < N; i++) {
    a[i] = i;
    b1[i] = 2 * i;
    b2[i] = 3 * i;
  }
  float *a_device, *b1_device, *b2_device, *c_device;
  cudaMalloc(&a_device, sizeof(a));
  cudaMalloc(&b1_device, sizeof(b1));
  cudaMalloc(&b2_device, sizeof(b2));
  cudaMalloc(&c_device, sizeof(c));

  cudaMemcpy(a_device, a, sizeof(a), cudaMemcpyHostToDevice);
  cudaMemcpy(b1_device, b1, sizeof(b1), cudaMemcpyHostToDevice);
  cudaMemcpy(b2_device, b2, sizeof(b2), cudaMemcpyHostToDevice);

  vector_add<<<1, N>>>(a_device, b1_device, b2_device,c_device, N);
  cudaMemcpy(c, c_device, sizeof(c), cudaMemcpyDeviceToHost);
  for (int i = 0; i < N; i++) {
    printf("%.2f\n", c[i]);
  }
}