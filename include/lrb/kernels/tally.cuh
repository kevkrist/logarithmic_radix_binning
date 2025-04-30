#pragma once

#include <cub/cub.cuh>
#include <cuda/std/iterator>

// With work functor
template <typename LogarithmicRadixBinner,
          int32_t B,
          int32_t BlockThreads,
          int32_t ItemsPerThread,
          typename InputIteratorT,
          typename CounterT,
          typename WorkFunctorT>
__global__ void
tally_kernel(InputIteratorT input, size_t size, WorkFunctorT work_functor, CounterT* bins)
{
  static_assert(BlockThreads >= B);
  using T            = cuda::std::iter_value_t<InputIteratorT>;
  using block_load_t = cub::BlockLoad<T, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_STRIPED>;

  static constexpr int32_t tile_items = BlockThreads * ItemsPerThread;
  const CounterT offset               = static_cast<CounterT>(blockIdx.x) * tile_items;
  const bool is_last_tile             = size <= offset + tile_items;
  const int32_t num_tile_items        = is_last_tile ? size - offset : tile_items;

  // Shared data
  __shared__ CounterT shared_bins[B];
  __shared__ typename block_load_t::TempStorage shared_load_storage; // 1B

  // Register data
  T thread_items[ItemsPerThread];

  if (is_last_tile)
  {
    // Load data
    block_load_t(shared_load_storage).Load(input + offset, thread_items, num_tile_items);

    // Increment bins
    LogarithmicRadixBinner::increment_shared_bins<BlockThreads, ItemsPerThread>(shared_bins,
                                                                                thread_items,
                                                                                num_tile_items,
                                                                                work_functor);
  }
  else
  {
    // Load data
    block_load_t(shared_load_storage).Load(input + offset, thread_items);

    // Increment bins
    LogarithmicRadixBinner::increment_shared_bins<BlockThreads, ItemsPerThread>(shared_bins,
                                                                                thread_items,
                                                                                work_functor);
  }
  __syncthreads();

  // Store bin data
  LogarithmicRadixBinner::update_global_bins(bins, shared_bins);
}

// With work values
template <typename LogarithmicRadixBinner,
          int32_t B,
          int32_t BlockThreads,
          int32_t ItemsPerThread,
          typename InputIteratorT,
          typename CounterT>
__global__ void tally_kernel(InputIteratorT input, size_t size, CounterT* bins)
{
  static_assert(BlockThreads >= B);
  using T            = cuda::std::iter_value_t<InputIteratorT>;
  using block_load_t = cub::BlockLoad<T, BlockThreads, ItemsPerThread, cub::BLOCK_LOAD_STRIPED>;

  static constexpr int32_t tile_items = BlockThreads * ItemsPerThread;
  const CounterT offset               = static_cast<CounterT>(blockIdx.x) * tile_items;
  const bool is_last_tile             = size <= offset + tile_items;
  const int32_t num_tile_items        = is_last_tile ? size - offset : tile_items;

  // Shared data
  __shared__ CounterT shared_bins[B];
  __shared__ typename block_load_t::TempStorage shared_load_storage; // 1B

  // Register data
  T thread_work[ItemsPerThread];

  if (is_last_tile)
  {
    // Load data
    block_load_t(shared_load_storage).Load(input + offset, thread_work, num_tile_items);

    // Increment bins
    LogarithmicRadixBinner::increment_shared_bins<BlockThreads, ItemsPerThread>(shared_bins,
                                                                                thread_work,
                                                                                num_tile_items);
  }
  else
  {
    // Load data
    block_load_t(shared_load_storage).Load(input + offset, thread_work);

    // Increment bins
    LogarithmicRadixBinner::increment_shared_bins<BlockThreads, ItemsPerThread>(shared_bins,
                                                                                thread_work);
  }
  __syncthreads();

  // Store bin data
  LogarithmicRadixBinner::update_global_bins(bins, shared_bins);
}