#include <vector>
#include <random>
#include <cuda.h>

__global__ void atomic_add_test(float* ptc_w, int* ptc_cell, float* grid_w, int n_ptc, int n_grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int n = tid; n < n_ptc; n += blockDim.x * gridDim.x) {
    atomicAdd(&grid_w[ptc_cell[n]], ptc_w[n]);
  }
}

int main(int argc, char** argv) {
  // Declare and allocate memory
  int N = 10000000;
  int N_grid = 100000;
  int ptc_per_cell = N / N_grid;
  float* p_ptc_w;
  int* p_ptc_cell;
  float* p_grid_w;

  cudaMalloc(&p_ptc_w, N * sizeof(float));
  cudaMalloc(&p_ptc_cell, N * sizeof(int));
  cudaMalloc(&p_grid_w, N_grid * sizeof(float));


  // Host pointers and initialization
  std::vector<float> v_ptc_w(N), v_grid_w(N_grid), v_grid_compare(N_grid);
  std::vector<int> v_ptc_cell(N);

  std::default_random_engine gen;
  std::uniform_real_distribution<float> dist_f(0.0f, 1.0f);
  for (int i = 0; i < N_grid; i++) {
    for (int j = 0; j < ptc_per_cell; j++) {
      int n = i * ptc_per_cell + j;
      v_ptc_w[n] = dist_f(gen);
      v_ptc_cell[n] = i;
    }
    v_grid_w[i] = 0.0f;
  }

  cudaMemcpy(p_ptc_w, v_ptc_w.data(), N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(p_ptc_cell, v_ptc_cell.data(), N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(p_grid_w, v_grid_w.data(), N_grid * sizeof(float), cudaMemcpyHostToDevice);

  // Run the atomicAdd kernel
  atomic_add_test<<<512, 512>>>(p_ptc_w, p_ptc_cell, p_grid_w, N, N_grid);
  cudaDeviceSynchronize();

  cudaMemcpy(v_grid_w.data(), p_grid_w, N_grid * sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(p_ptc_w);
  cudaFree(p_ptc_cell);
  cudaFree(p_grid_w);

  return 0;
}
