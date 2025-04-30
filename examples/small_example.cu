#define CUB_STDERR

#include <cub/cub.cuh>
#include <iostream>
#include <lrb/lrb.cuh>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

constexpr int32_t b                = 32;
constexpr int32_t block_threads    = 32;
constexpr int32_t items_per_thread = 2;
constexpr int32_t dist_max         = 10000;
constexpr size_t num_items =
  (items_per_thread + 1) * block_threads + 7; // Introduce some irregularity

int32_t main()
{
  using LrbT = LogarithmicRadixBinner<uint32_t, b>;

  // Populate host items with random values
  thrust::host_vector<uint32_t> items(num_items);
  std::mt19937 rng(std::random_device{}());
  std::uniform_int_distribution<uint32_t> dist(0, dist_max);
  for (auto& item : items)
  {
    item = dist(rng);
  }

  // Print input
  std::cout << "INPUT:\n";
  for (size_t i = 0; i < items.size(); ++i)
  {
    std::cout << "[" << i << "]: " << items[i] << "\n";
  }

  // Copy data to device (assume item value = work value)
  thrust::device_vector<uint32_t> items_d(num_items);
  thrust::device_vector<uint32_t> work_d(num_items);
  thrust::device_vector<uint32_t> shuffled_items_d(num_items);
  items_d = items;
  work_d  = items;

  // Do LRB
  size_t temp_storage_bytes = 0;
  cudaError_t err           = LrbT::execute<block_threads, items_per_thread>(nullptr,
                                                                   temp_storage_bytes,
                                                                   items_d.begin(),
                                                                   work_d.begin(),
                                                                   shuffled_items_d.begin(),
                                                                   num_items);
  CubDebugExit(err);
  thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
  err =
    LrbT::execute<block_threads, items_per_thread>(thrust::raw_pointer_cast(temp_storage.data()),
                                                   temp_storage_bytes,
                                                   items_d.begin(),
                                                   work_d.begin(),
                                                   shuffled_items_d.begin(),
                                                   num_items);
  CubDebugExit(err);
  CubDebugExit(cudaDeviceSynchronize());

  // Copy results back to host
  thrust::host_vector<uint32_t> shuffled_items(shuffled_items_d);

  // Print output
  std::cout << "\n\nOUTPUT:\n";
  for (size_t i = 0; i < shuffled_items.size(); ++i)
  {
    std::cout << "[" << i << "]: " << shuffled_items[i] << "\n";
  }

  return EXIT_SUCCESS;
}