#include <assert.h>
#include <cub/block/block_reduce.cuh>

#include "caffe2/core/context_gpu.h"
#include "kl_op.h"

namespace caffe2 {

namespace {

__device__ float sigmoid_xent_forward(float lgt, float tgt) {
  return tgt * log(1. / tgt / (1. + exp(-lgt)));
}

__device__ float sigmoid_xent_backward(float lgt, float tgt) {
  return tgt * (1. - 1. / (1. + exp(-lgt)));
}

__global__ void BalanceWSLKernel(const int outer_size, const int inner_size,
                                 const float* logits_ptr,
                                 const float* targets_ptr,
                                 const float ignore_value, float* count_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float pos = 0;
  float neg = 0;
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    if (targets_ptr[in_idx] == ignore_value) {
      continue;
    }
    if (targets_ptr[in_idx] > 0.5) {
      pos += 1;
    } else {
      neg += 1;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float pos_sum = BlockReduce(temp_storage).Sum(pos);

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce2;
  __shared__ typename BlockReduce2::TempStorage temp_storage2;
  float neg_sum = BlockReduce2(temp_storage2).Sum(neg);

  if (threadIdx.x == 0) {
    count_ptr[i * 2] = pos_sum;
    count_ptr[i * 2 + 1] = neg_sum;
  }
}

__global__ void KLKernel(const int outer_size, const int inner_size,
                         const float* logits_ptr, const float* targets_ptr,
                         const float* count_ptr, const float ignore_value,
                         float* out_ptr) {
  int i = blockIdx.x;
  int last_idx = (i + 1) * inner_size;
  float value = 0;
  float pos = count_ptr[i * 2];
  float neg = count_ptr[i * 2 + 1];
  for (int in_idx = i * inner_size + threadIdx.x; in_idx < last_idx;
       in_idx += blockDim.x) {
    if (targets_ptr[in_idx] == ignore_value) {
      continue;
    }
    if (targets_ptr[in_idx] > 0.5) {
      value +=
          sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) / pos;
    } else {
      value +=
          sigmoid_xent_forward(logits_ptr[in_idx], targets_ptr[in_idx]) / neg;
    }
  }

  typedef cub::BlockReduce<float, CAFFE_CUDA_NUM_THREADS> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;
  float sum = BlockReduce(temp_storage).Sum(value);
  if (threadIdx.x == 0) {
    out_ptr[i] = -sum;
  }
}

__global__ void KLGradientKernel(const int outer_size, const int inner_size,
                                 const float* g_ptr, const float* logits_ptr,
                                 const float* targets_ptr,
                                 const float* count_ptr,
                                 const float ignore_value, float* out_ptr) {
  CUDA_1D_KERNEL_LOOP(in_idx, outer_size * inner_size) {
    int i = in_idx / inner_size;
    if (targets_ptr[in_idx] == ignore_value) {
      out_ptr[in_idx] = 0.0;
      continue;
    }
    // auto g_factor = -g_ptr[i] / inner_size;
    float g_factor;
    float count;
    if (targets_ptr[in_idx] > 0.5) {
      count = count_ptr[i * 2];
    } else {
      count = count_ptr[i * 2 + 1];
    }
    if (count > 0) {
      g_factor = -g_ptr[i] / count;
    } else {
      g_factor = 0;
    }
    out_ptr[in_idx] = g_factor * sigmoid_xent_backward(logits_ptr[in_idx],
                                                       targets_ptr[in_idx]);
  }
}
}  // namespace

template <>
bool KLOp<float, CUDAContext>::RunOnDevice() {
  auto& logits = Input(0);
  auto& targets = Input(1);
  CAFFE_ENFORCE_EQ(logits.dims(), targets.dims());
  const auto inner_size =
      logits.dim() > 0 ? logits.dim32(2) * logits.dim32(3) : 1;
  const auto outer_size = logits.numel() / inner_size;

  auto* out = Output(0);
  auto* count = Output(1);
  if (logits.dim() == 0) {
    out->Resize(std::vector<int64_t>{});
    count->Resize(std::vector<int64_t>{});
  } else {
    std::vector<int64_t> dims(logits.dims().begin(), logits.dims().end() - 2);
    out->Resize(dims);
    dims.push_back(2);
    count->Resize(dims);
  }
  auto* out_ptr = out->mutable_data<float>();
  auto* count_ptr = count->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();

  if (logits.numel() <= 0) {
    // nothing to do, not even launching kernel
    return true;
  }

  BalanceWSLKernel<<<outer_size, CAFFE_CUDA_NUM_THREADS, 0,
                     context_.cuda_stream()>>>(outer_size, inner_size,
                                               logits_ptr, targets_ptr,
                                               ignore_value_, count_ptr);

  KLKernel<<<outer_size, CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      outer_size, inner_size, logits_ptr, targets_ptr, count_ptr, ignore_value_,
      out_ptr);
  return true;
}

template <>
bool KLGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& g = Input(0);
  auto& logits = Input(1);
  auto& targets = Input(2);
  auto& count = Input(3);
  CAFFE_ENFORCE_EQ(logits.dims(), targets.dims());
  const auto inner_size =
      logits.dim() > 0 ? logits.dim32(2) * logits.dim32(3) : 1;
  const auto outer_size = logits.numel() / inner_size;
  CAFFE_ENFORCE_EQ(g.numel(), outer_size);

  auto* out = Output(0);
  out->ResizeLike(logits);
  auto* out_ptr = out->mutable_data<float>();

  auto* logits_ptr = logits.data<float>();
  auto* targets_ptr = targets.data<float>();
  auto* g_ptr = g.data<float>();
  auto* count_ptr = count.data<float>();

  KLGradientKernel<<<CAFFE_GET_BLOCKS(outer_size * inner_size),
                     CAFFE_CUDA_NUM_THREADS, 0, context_.cuda_stream()>>>(
      outer_size, inner_size, g_ptr, logits_ptr, targets_ptr, count_ptr,
      ignore_value_, out_ptr);
  return true;
}

REGISTER_CUDA_OPERATOR(KL, KLOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(KLGradient, KLGradientOp<float, CUDAContext>);

}  // namespace caffe2
