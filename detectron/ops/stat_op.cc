#include <functional>

#include "stat_op.h"

namespace caffe2 {

namespace {}  // namespace

using namespace std::placeholders;

OPERATOR_SCHEMA(Stat)
    .NumInputs(1)
    .NumOutputs(1)
    .SetDoc(R"DOC(
)DOC")
    .Arg("debug_info", "(bool) default to false")
    .Arg("prefix",
         "(string) default to "
         "")
    .Input(0, "in", "input tensor")
    .Output(0, "useless", "output tensor");

namespace {

NO_GRADIENT(Stat);

}  // namespace

}  // namespace caffe2
