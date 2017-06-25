/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/framework/op_kernel.h"
#include "tiny_dnn/core/kernels/recurrent_cell_op_internal.h"

namespace tiny_dnn {

class RecurrentCellOp : public core::OpKernel {
 public:
  explicit RecurrentCellOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->recurrent_cell();

    // incomimg/outcoming data
    // TODO(Randl): Remove once layers forward and backward by themself.
    const Tensor<float_t> in_data(context.input(0));
    const Tensor<float_t> prev_h(context.input(1));
    const Tensor<float_t> U(context.input(2));
    const Tensor<float_t> W(context.input(3));
    const Tensor<float_t> V(context.input(4));
    const Tensor<float_t> bias = params.has_bias_ ? Tensor<float_t>(context.input(5)) : Tensor<float_t>();
    const Tensor<float_t> c = params.has_bias_ ? Tensor<float_t>(context.input(6)) : Tensor<float_t>();
    Tensor<float_t> out_data(context.output(0));
    Tensor<float_t> next_h(context.output(1));

    // initialize outputs
    out_data.fill(0);
    next_h.fill(0);

    // call the algorithm depending  on the selected engine type

    const core::backend_t engine = context.engine();

    if (engine == core::backend_t::internal || engine == core::backend_t::avx) {
      kernels::recurrent_cell_op_internal(
        in_data, prev_h, U, W, V,
        bias, c, out_data, next_h, params,
        context.parallelize());
      // TODO(Randl): Remove once layers forward and backward by themself.
      context.output(0) = out_data.toTensor();
    } else {
      throw nn_error("Not supported engine: " + to_string(engine));
    }
  }
};

}  // namespace tiny_dnn
