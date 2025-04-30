#define CUB_STDERR

#include <cub/cub.cuh>
#include <fstream>
#include <iostream>
#include <lrb/lrb.cuh>
#include <sstream>
#include <string>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/// CONFIG ///
const std::string profile_data        = "/home/ubuntu/lrb/examples/project/profile_data.csv";
const std::string shuffled_input_data = "/home/ubuntu/lrb/examples/project/shuffled_input_data.csv";
constexpr int32_t b                   = 32;
constexpr int32_t block_threads       = 128;
constexpr int32_t items_per_thread    = 4;
/// END CONFIG ///

void parse_csv(const std::string& filename,
               thrust::host_vector<int32_t>& input,
               thrust::host_vector<uint32_t>& active_cycles)
{
  std::ifstream file(filename);
  std::string line;

  if (!file.is_open())
  {
    throw std::runtime_error("Failed to open file.");
  }

  // Skip the header
  std::getline(file, line);

  while (std::getline(file, line))
  {
    std::istringstream ss(line);
    std::string token;

    // Skip Thread column
    std::getline(ss, token, ',');

    // Input
    std::getline(ss, token, ',');
    input.push_back(std::stoi(token));

    // Skip Output column
    std::getline(ss, token, ',');

    // Active Cycles
    std::getline(ss, token, ',');
    active_cycles.push_back(static_cast<unsigned int>(std::stoul(token)));

    // Skip the rest of the line...
  }
}

void write_csv(const std::string& filename, const thrust::host_vector<int32_t>& shuffled_input)
{
  std::ofstream out(filename);
  if (!out.is_open())
  {
    throw std::runtime_error("Failed to open output file.");
  }

  // Write header
  out << "Index,Input\n";

  // Write shuffled data
  for (size_t i = 0; i < shuffled_input.size(); ++i)
  {
    out << i << "," << shuffled_input[i] << "\n";
  }
}

int32_t main()
{
  using LrbT = LogarithmicRadixBinner<uint32_t, b>;

  // Initialize host vectors
  thrust::host_vector<int32_t> input{};
  thrust::host_vector<uint32_t> active_cycles{};

  try
  {
    // Get the input and active cycles
    parse_csv(profile_data, input, active_cycles);

    // Copy data to device
    thrust::device_vector<int32_t> input_d(input);
    thrust::device_vector<uint32_t> work_d(active_cycles);
    thrust::device_vector<int32_t> shuffled_input_d(input.size());

    // Do LRB
    size_t temp_storage_bytes = 0;
    cudaError_t err           = LrbT::execute<block_threads, items_per_thread>(nullptr,
                                                                     temp_storage_bytes,
                                                                     input_d.begin(),
                                                                     work_d.begin(),
                                                                     shuffled_input_d.begin(),
                                                                     input.size());
    CubDebugExit(err);
    thrust::device_vector<uint8_t> temp_storage(temp_storage_bytes);
    err =
      LrbT::execute<block_threads, items_per_thread>(thrust::raw_pointer_cast(temp_storage.data()),
                                                     temp_storage_bytes,
                                                     input_d.begin(),
                                                     work_d.begin(),
                                                     shuffled_input_d.begin(),
                                                     input.size());
    CubDebugExit(err);
    CubDebugExit(cudaDeviceSynchronize());

    // Write shuffled data
    thrust::host_vector<int32_t> shuffled_input(shuffled_input_d);
    write_csv(shuffled_input_data, shuffled_input);
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << e.what() << "\n";
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
