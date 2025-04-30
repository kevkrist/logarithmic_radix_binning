#pragma once

#include "kernels/shuffle.cuh"
#include "kernels/tally.cuh"
#include <cub/cub.cuh>
#include <cuda/atomic>
#include <cuda/std/bit>
#include <type_traits>

template <typename CounterT, int32_t B>
struct LogarithmicRadixBinner
{
  static_assert(B == 32 || B == 64);
  static_assert(std::is_integral_v<CounterT> && std::is_unsigned_v<CounterT>);
  using LrbT = LogarithmicRadixBinner<CounterT, B>;

  // With work functor
  template <int32_t BlockThreads,
            int32_t ItemsPerThread,
            typename InputIteratorT,
            typename OutputIteratorT,
            typename WorkFunctorT>
  __host__ static cudaError_t execute(uint8_t* temp_storage,
                                      size_t& temp_storage_bytes,
                                      InputIteratorT input,
                                      OutputIteratorT output,
                                      size_t size,
                                      WorkFunctorT&& work_functor)
  {
    static_assert(BlockThreads >= B);
    constexpr int32_t tile_items = BlockThreads * ItemsPerThread;
    cudaError_t err              = cudaSuccess;
    size_t bin_bytes             = sizeof(CounterT) * B;

    // If temp_storage is null, calculate the number of temporary storage bytes needed for LRB
    if (temp_storage == nullptr)
    {
      // Prefix sum storage
      CounterT* dummy_bins = nullptr;
      err = CubDebug(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, dummy_bins, B));

      // Bin storage
      temp_storage_bytes += bin_bytes;
      return err;
    }

    // Assume temp_storage has been correctly allocated
    CounterT* bins            = reinterpret_cast<CounterT*>(temp_storage);
    uint8_t* scan_storage     = temp_storage + bin_bytes;
    size_t scan_storage_bytes = temp_storage_bytes - bin_bytes;

    // Increment bins
    tally_kernel<LrbT, B, BlockThreads, ItemsPerThread>
      <<<cuda::ceil_div(size, static_cast<size_t>(tile_items)), BlockThreads>>>(input,
                                                                                size,
                                                                                work_functor,
                                                                                bins);
    err = CubDebug(cudaGetLastError());
    if (err != cudaSuccess)
    {
      return err;
    }

    // Prefix sum bins
    err = CubDebug(cub::DeviceScan::ExclusiveSum(scan_storage, scan_storage_bytes, bins, B));
    if (err != cudaSuccess)
    {
      return err;
    }

    // Shuffle data
    shuffle_kernel<LrbT>
      <<<cuda::ceil_div(size, static_cast<size_t>(BlockThreads)), BlockThreads>>>(input,
                                                                                  output,
                                                                                  size,
                                                                                  bins,
                                                                                  work_functor);
    return CubDebug(cudaGetLastError());
  }

  // Without work functor
  template <int32_t BlockThreads,
            int32_t ItemsPerThread,
            typename InputIteratorT,
            typename WorkIteratorT,
            typename OutputIteratorT>
  __host__ static cudaError_t execute(uint8_t* temp_storage,
                                      size_t& temp_storage_bytes,
                                      InputIteratorT input,
                                      WorkIteratorT work,
                                      OutputIteratorT output,
                                      size_t size)
  {
    constexpr int32_t tile_items = BlockThreads * ItemsPerThread;
    static_assert(BlockThreads >= B);
    cudaError_t err  = cudaSuccess;
    size_t bin_bytes = sizeof(CounterT) * B;

    // If temp_storage is null, calculate the number of temporary storage bytes needed for LRB
    // TODO: refactor
    if (temp_storage == nullptr)
    {
      // Prefix sum storage
      CounterT* dummy_bins = nullptr;
      err = CubDebug(cub::DeviceScan::ExclusiveSum(nullptr, temp_storage_bytes, dummy_bins, B));

      // Bin storage
      temp_storage_bytes += bin_bytes;
      return err;
    }

    // Assume temp_storage has been correctly allocated
    CounterT* bins            = reinterpret_cast<CounterT*>(temp_storage);
    uint8_t* scan_storage     = temp_storage + bin_bytes;
    size_t scan_storage_bytes = temp_storage_bytes - bin_bytes;

    // Increment bins
    tally_kernel<LrbT, B, BlockThreads, ItemsPerThread>
      <<<cuda::ceil_div(size, static_cast<size_t>(tile_items)), BlockThreads>>>(work, size, bins);
    err = CubDebug(cudaGetLastError());
    if (err != cudaSuccess)
    {
      return err;
    }

    // Prefix sum bins
    err = CubDebug(cub::DeviceScan::ExclusiveSum(scan_storage, scan_storage_bytes, bins, B));
    if (err != cudaSuccess)
    {
      return err;
    }

    // Shuffle data
    shuffle_kernel<LrbT>
      <<<cuda::ceil_div(size, static_cast<size_t>(BlockThreads)), BlockThreads>>>(input,
                                                                                  work,
                                                                                  output,
                                                                                  size,
                                                                                  bins);
    return CubDebug(cudaGetLastError());
  }

  // Increment bins for a full tile
  template <int32_t BlockThreads, int32_t ItemsPerThread, typename T, typename WorkFunctorT>
  static __device__ __forceinline__ void
  increment_shared_bins(CounterT (&shared_bins)[B],
                        const T (&thread_items)[ItemsPerThread],
                        WorkFunctorT&& work_functor)
  {
#pragma unroll ItemsPerThread
    for (int32_t i = 0; i < ItemsPerThread; ++i)
    {
      const auto log_work = floor_log_2(work_functor(thread_items[i]));
      auto shared_bin_ref =
        cuda::atomic_ref<CounterT, cuda::thread_scope_block>(*(shared_bins + log_work));
      shared_bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
  }
  template <int32_t BlockThreads, int32_t ItemsPerThread, typename T>
  static __device__ __forceinline__ void
  increment_shared_bins(CounterT (&shared_bins)[B], const T (&thread_work)[ItemsPerThread])
  {
#pragma unroll ItemsPerThread
    for (int32_t i = 0; i < ItemsPerThread; ++i)
    {
      const auto log_work = floor_log_2(thread_work[i]);
      auto shared_bin_ref =
        cuda::atomic_ref<CounterT, cuda::thread_scope_block>(*(shared_bins + log_work));
      shared_bin_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
  }

  // Increment bins for a partial tile
  template <int32_t BlockThreads, int32_t ItemsPerThread, typename T, typename WorkFunctorT>
  static __device__ __forceinline__ void
  increment_shared_bins(CounterT (&shared_bins)[B],
                        const T (&thread_items)[ItemsPerThread],
                        int32_t num_tile_items,
                        WorkFunctorT&& work_functor)
  {
#pragma unroll ItemsPerThread
    for (int32_t i = 0; i < ItemsPerThread; ++i)
    {
      if (threadIdx.x + BlockThreads * i < num_tile_items)
      {
        const auto log_work = floor_log_2(work_functor(thread_items[i]));
        auto shared_bin_ref =
          cuda::atomic_ref<CounterT, cuda::thread_scope_block>(*(shared_bins + log_work));
        shared_bin_ref.fetch_add(1, cuda::memory_order_relaxed);
      }
    }
  }
  template <int32_t BlockThreads, int32_t ItemsPerThread, typename T>
  static __device__ __forceinline__ void
  increment_shared_bins(CounterT (&shared_bins)[B],
                        const T (&thread_work)[ItemsPerThread],
                        int32_t num_tile_items)
  {
#pragma unroll ItemsPerThread
    for (int32_t i = 0; i < ItemsPerThread; ++i)
    {
      if (threadIdx.x + BlockThreads * i < num_tile_items)
      {
        const auto log_work = floor_log_2(thread_work[i]);
        auto shared_bin_ref =
          cuda::atomic_ref<CounterT, cuda::thread_scope_block>(*(shared_bins + log_work));
        shared_bin_ref.fetch_add(1, cuda::memory_order_relaxed);
      }
    }
  }

  // Update global bins
  static __device__ __forceinline__ void update_global_bins(CounterT* bins,
                                                            const CounterT (&shared_bins)[B])
  {
    if (threadIdx.x < B)
    {
      auto global_bin_ref =
        cuda::atomic_ref<CounterT, cuda::thread_scope_device>(*(bins + threadIdx.x));
      global_bin_ref.fetch_add(shared_bins[threadIdx.x], cuda::memory_order_relaxed);
    }
  }

  // Compute log_2 safely
  template <typename T>
  static __device__ __forceinline__ T floor_log_2(T val)
  {
    static_assert(std::is_integral_v<T> && std::is_unsigned_v<T>);
    return val == 0 ? 0 : cuda::std::bit_width(val) - 1;
  }
};