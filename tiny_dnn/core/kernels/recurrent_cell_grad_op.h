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

class RecurrentCellGradOp : public core::OpKernel {
 public:
  explicit RecurrentCellGradOp(const core::OpKernelConstruction &context)
    : core::OpKernel(context) {}

  void compute(core::OpKernelContext &context) override {
    auto params = OpKernel::params_->recurrent_cell();
    // incoming/outcoming data
    // TODO(Randl): Remove once layers forward and backward by themself.
    const Tensor<float_t> prev_out(context.input(0));
    const Tensor<float_t> h(context.input(1));
    const Tensor<float_t> U(context.input(2));
    const Tensor<float_t> W(context.input(3));
    const Tensor<float_t> V(context.input(4));
    Tensor<float_t> dU(context.input_grad(2));
    Tensor<float_t> dW(context.input_grad(3));
    Tensor<float_t> dV(context.input_grad(4));
    Tensor<float_t> db = params.has_bias_ ? Tensor<float_t>(context.input_grad(5)) : Tensor<float_t>();
    Tensor<float_t> dc = params.has_bias_ ? Tensor<float_t>(context.input_grad(6)) : Tensor<float_t>();
    Tensor<float_t> prev_output_delta(context.input_grad(0));
    Tensor<float_t> prev_state_delta(context.input_grad(1));
    Tensor<float_t> curr_output_delta(context.output_grad(0));
    Tensor<float_t> curr_state_delta(context.output_grad(1));
    const Tensor<float_t> out_state(context.output(1));

    // initialize outputs
    prev_output_delta.fill(0);
    prev_state_delta.fill(0);

    // call the algorithm depending on the selected engine type

    const core::backend_t engine = context.engine();

    kernels::recurrent_cell_op_internal(
      prev_out, h, U, W, V, dU, dW, dV, db,
      dc, curr_output_delta, curr_state_delta,
      prev_output_delta, prev_state_delta, out_state, params,
      context.parallelize());
      context.input_grad(0) = prev_output_delta.toTensor();
      context.input_grad(1) = prev_state_delta.toTensor();
      context.input_grad(2) = dU.toTensor();
      context.input_grad(3) = dW.toTensor();
      context.input_grad(4) = dV.toTensor();
      if (params.has_bias_) {
        context.input_grad(5) = db.toTensor();
        context.input_grad(6) = dc.toTensor();
      }
  }
};

}  // namespace tiny_dnn
