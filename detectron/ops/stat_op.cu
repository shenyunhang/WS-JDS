#include <functional>

#include "caffe2/core/context_gpu.h"
#include "stat_op.h"

namespace caffe2 {

namespace {}  // namespace

template <>
bool StatOp<float, CUDAContext>::RunOnDevice() {
  const auto& I = Input(0);
  auto* S = Output(0);
  if (init_) {
    S->ResizeLike(I);
    math::Set<float, CUDAContext>(S->numel(), 0.f, S->mutable_data<float>(),
                                  &context_);
    init_ = false;
  }
  math::Add<float, CUDAContext>(I.size(), I.data<float>(), S->data<float>(),
                                S->mutable_data<float>(), &context_);

  cur_iter_++;
  if (cur_iter_ % display_ == 0 || cur_iter_ == 1) {
    Tensor mean_cpu = Tensor(caffe2::CPU);
    mean_cpu.ResizeLike(*S);

    mean_cpu.CopyFrom(*S, false);
    context_.FinishDeviceComputation();

    const float* mean_cpu_data = mean_cpu.data<float>();

    std::cout << prefix_ << " Stat #iter_: " << cur_iter_;
    for (int i = 0; i < mean_cpu.size(); i++) {
      std::cout << "  " << mean_cpu_data[i];
    }
    std::cout << std::endl;
    init_ = true;
  }
  return true;
}

REGISTER_CUDA_OPERATOR(Stat, StatOp<float, CUDAContext>);

}  // namespace caffe2
