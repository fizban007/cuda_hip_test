#include <vector>
#include <random>
#include <hip/hip_runtime.h>

struct A {
  int a, b;
  float c;
};

__constant__ A dev_a;

__global__ void atomic_add_test(float* ptc_w, int* ptc_cell, float* grid_w, int n_ptc, int n_grid) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;

  for (int n = tid; n < n_ptc; n += blockDim.x * gridDim.x) {
    atomicAdd(&grid_w[ptc_cell[n]], ptc_w[n]);
  }
}

int main(int argc, char *argv[]) {
  // Declare and allocate memory
  int N = 10000000;
  int N_grid = 100000;
  int ptc_per_cell = N / N_grid;
  float* p_ptc_w;
  int* p_ptc_cell;
  float* p_grid_w;

  hipMalloc(&p_ptc_w, N * sizeof(float));
  hipMalloc(&p_ptc_cell, N * sizeof(int));
  hipMalloc(&p_grid_w, N_grid * sizeof(float));


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

  hipMemcpy(p_ptc_w, v_ptc_w.data(), N * sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(p_ptc_cell, v_ptc_cell.data(), N * sizeof(int), hipMemcpyHostToDevice);
  hipMemcpy(p_grid_w, v_grid_w.data(), N_grid * sizeof(float), hipMemcpyHostToDevice);

  // Run the atomicAdd kernel
  atomic_add_test<<<512, 512>>>(p_ptc_w, p_ptc_cell, p_grid_w, N, N_grid);
  hipDeviceSynchronize();

  hipMemcpy(v_grid_w.data(), p_grid_w, N_grid * sizeof(float), hipMemcpyDeviceToHost);

  // Free memory
  hipFree(p_ptc_w);
  hipFree(p_ptc_cell);
  hipFree(p_grid_w);

  A host_a{1, 1, 1.0};
  // hipMemcpyToSymbol(HIP_SYMBOL(&dev_a), &host_a, sizeof(A), 0, hipMemcpyHostToDevice);
  hipMemcpyToSymbol(HIP_SYMBOL(dev_a), &host_a, sizeof(A), 0, hipMemcpyHostToDevice);

  return 0;
}
