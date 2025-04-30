#pragma once

#include <cuda/atomic>
#include <cuda/std/bit>
#include <cuda/std/iterator>

// TODO: there is a more efficient way to do this, but postponing for now

// With work functor
template <typename LogarithmicRadixBinner,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename CounterT,
          typename WorkFunctorT>
__global__ void shuffle_kernel(InputIteratorT input,
                               OutputIteratorT output,
                               size_t size,
                               CounterT* prefix_bins,
                               WorkFunctorT work_functor)
{
  using T            = cuda::std::iter_value_t<InputIteratorT>;
  const size_t g_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (g_idx < size)
  {
    // Load input item
    const T input_item = input[g_idx];

    // Compute work
    const auto log_work = LogarithmicRadixBinner::floor_log_2(work_functor(input_item));

    // Increment prefix bin
    auto prefix_bin_ref =
      cuda::atomic_ref<CounterT, cuda::thread_scope_device>(*(prefix_bins + log_work));
    const auto out_idx = prefix_bin_ref.fetch_add(1, cuda::memory_order_relaxed);

    // Store in output iterator
    output[out_idx] = input_item;
  }
}

// Without work functor
template <typename LogarithmicRadixBinner,
          typename InputIteratorT,
          typename WorkIteratorT,
          typename OutputIteratorT,
          typename CounterT>
__global__ void shuffle_kernel(InputIteratorT input,
                               WorkIteratorT work,
                               OutputIteratorT output,
                               size_t size,
                               CounterT* prefix_bins)
{
  using T            = cuda::std::iter_value_t<InputIteratorT>;
  using W            = cuda::std::iter_value_t<WorkIteratorT>;
  const size_t g_idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

  if (g_idx < size)
  {
    // Load work, input item
    const W work_value = work[g_idx];
    const T input_item = input[g_idx];

    // Compute work
    const auto log_work = LogarithmicRadixBinner::floor_log_2(work_value);

    // Increment prefix bin
    auto prefix_bin_ref =
      cuda::atomic_ref<CounterT, cuda::thread_scope_device>(*(prefix_bins + log_work));
    const auto out_idx = prefix_bin_ref.fetch_add(1, cuda::memory_order_relaxed);

    // Store in output iterator
    output[out_idx] = input_item;
  }
}