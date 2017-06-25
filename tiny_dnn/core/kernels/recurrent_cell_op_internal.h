/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/recurrent_cell_params.h"

namespace tiny_dnn {
namespace kernels {

/**
 * Forward propogation for recurrent cell layer with internal backend
 * @param in_data
 * @param prev_h
 * @param U
 * @param W
 * @param V
 * @param bias
 * @param c
 * @param out_data
 * @param out_h
 * @param params
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3, typename S4, typename S5, typename S6, typename S7, typename S8, typename S9>
inline void recurrent_cell_op_internal(const Tensor<float_t, S1> &in_data,
                                       const Tensor<float_t, S2> &prev_h,
                                       const Tensor<float_t, S3> &U,
                                       const Tensor<float_t, S4> &W,
                                       const Tensor<float_t, S5> &V,
                                       const Tensor<float_t, S6> &bias,
                                       const Tensor<float_t, S7> &c,
                                       Tensor<float_t, S8> &out_data,
                                       Tensor<float_t, S9> &out_h,
                                       recurrent_cell_params &params,
                                       const bool layer_parallelize) {

  size_t out_size = out_data.shape()[1], in_size = in_data.shape()[1];

  for_i(layer_parallelize, in_data.shape()[0], [&](size_t sample) {
    /*const vec_t &in         = in_data[sample];
    const vec_t &prev_state = prev_h[sample];
    vec_t &out              = out_data[sample];
    vec_t &next_state       = out_h[sample];
    */
    for (size_t o = 0; o < out_size; o++) {
      float_t next_state_ = 0;

      // W * h(t-1)
      for (size_t o_2 = 0; o_2 < out_size; o_2++) {
        next_state_ += W.host_at(0, o_2 * out_size + o) * prev_h.host_at(sample, o_2);
      }

      // U*x(t)
      for (size_t i = 0; i < in_size; i++) {
        next_state_ += U.host_at(0, i * out_size + o) * in_data.host_at(sample, i);
      }

      if (params.has_bias_) {
        next_state_ += bias.host_at(0, o);
      }
      out_h.host_at(sample, o) = next_state_;
    }
    // TODO(prlz77) use tensors when activations allow it
    params.activation_->forward_activation(out_h[sample].toVec(), out_h[sample].toVec());

    // V matrix is out_size_ x out_size_
    for (size_t o = 0; o < out_size; o++) {
      float_t out_ = 0;
      for (size_t o_2 = 0; o_2 < out_size; o_2++) {
        out_ += V.host_at(0, o_2 * out_size + o) * out_h.host_at(o_2);
      }

      if (params.has_bias_) {
        out_ += c.host_at(0, o);
      }
      out_data.host_at(sample, o) = out_;
    }

  });
}

/**
 * Forward propogation for recurrent cell layer with internal backend
 * @param prev_out
 * @param prev_h
 * @param U
 * @param W
 * @param V
 * @param dU
 * @param dW
 * @param dV
 * @param db
 * @param dc
 * @param curr_output_delta
 * @param curr_state_delta
 * @param prev_output_delta
 * @param prev_state_delta
 * @param out_h
 * @param params
 * @param layer_parallelize
 */
template <typename S1, typename S2, typename S3, typename S4, typename S5, typename S6, typename S7, typename S8, typename S9, typename S10, typename S11, typename S12, typename S13, typename S14, typename S15>
inline void recurrent_cell_op_internal(const Tensor<float_t, S1> &prev_out,
                                       const Tensor<float_t, S2> &prev_h,
                                       const Tensor<float_t, S3> &U,
                                       const Tensor<float_t, S4> &W,
                                       const Tensor<float_t, S5> &V,
                                       Tensor<float_t, S6> &dU,
                                       Tensor<float_t, S7> &dW,
                                       Tensor<float_t, S8> &dV,
                                       Tensor<float_t, S9> &db,
                                       Tensor<float_t, S10> &dc,
                                       const Tensor<float_t, S11> &curr_output_delta,
                                       Tensor<float_t, S12> &curr_state_delta,
                                       Tensor<float_t, S13> &prev_output_delta,
                                       Tensor<float_t, S14> &prev_state_delta,
                                       const Tensor<float_t, S15> &out_h,
                                       const recurrent_cell_params &params,
                                       const bool layer_parallelize) {
  for (size_t sample = 0; sample < prev_out.size(); sample++) {
    /*
    const vec_t &prev_out_          = prev_out[sample];
    const vec_t &prev_h_            = prev_h[sample];
    vec_t &dU_                      = dU[sample];
    vec_t &dW_                      = dW[sample];
    vec_t &dV_                      = dV[sample];
    vec_t &db_                      = db[sample];
    vec_t &dc_                      = dc[sample];
    const vec_t &curr_output_delta_ = curr_output_delta[sample];
    vec_t &curr_state_delta_        = curr_state_delta[sample];
    vec_t &prev_output_delta_       = prev_output_delta[sample];
    vec_t &prev_state_delta_        = prev_state_delta[sample];
    const vec_t &out_h_             = out_h[sample];
    */
    const size_t out_size = out_h.shape()[1], in_size = prev_out.shape()[1];
    // from output to h
    for (size_t o = 0; o < out_size; o++) {
      // propagate delta from output to h.
      curr_state_delta.host_at(sample, o) += vectorize::dot(
        curr_output_delta.host_pointer(sample, 0), V.host_pointer(o * out_size), out_size);
    }

    // h'(t)
    params.activation_->backward_activation(prev_h.host_at(sample), out_h.host_at(sample), curr_state_delta.host_at(sample),
                                            curr_state_delta.host_at(sample));

    // \delta h(t) -W-> h(t-1)
    for (size_t o = 0; o < out_size; o++) {
      prev_state_delta.host_at(sample, o) += vectorize::dot(
        curr_state_delta.host_pointer(sample, 0), W.host_pointer(o * out_size), out_size);
    }

    // \delta h(t) -U-> \delta x(t)
    for (size_t i = 0; i < in_size; i++) {
      prev_output_delta.host_at(sample, i) += vectorize::dot(
        curr_state_delta.host_pointer(sample, 0), U.host_pointer(i * out_size), out_size);
    }

    for_(layer_parallelize, 0, size_t(out_size),
      [&](const blocked_range &r) {
        // accumulate weight-step using delta
        // dW[c * out_size + i] += current_delta[i] * prev_out[o]
        const size_t begin  = r.begin();
        const size_t end    = r.end();
        const size_t stride = end - begin;
        for (size_t o = 0; o < out_size; o++) {
          vectorize::muladd(curr_output_delta.host_pointer(sample, begin), out_h.host_pointer(sample, o), stride,
                            dV.host_pointer(sample, o * out_size + begin));
        }

        if (params.has_bias_) {
          // vec_t& dc;
          for (size_t o = begin; o < end; o++) {
            dc.host_at(sample, o) += curr_output_delta.host_at(sample, o);
          }
        }

        for (size_t o = 0; o < out_size; o++) {
          vectorize::muladd(curr_state_delta.host_pointer(sample, begin), prev_h.host_pointer(sample, o), stride,
                            dW.host_pointer(sample, o * out_size + begin));
        }

        for (size_t i = 0; i < in_size; i++) {
          vectorize::muladd(curr_state_delta.host_pointer(sample, begin), prev_out.host_pointer(sample, i),
                            stride, dU.host_pointer(sample, i * out_size + begin));
        }

        if (params.has_bias_) {
          // vec_t& db;
          for (size_t o = begin; o < end; o++) {
            db.host_at(sample, o) += curr_state_delta.host_at(sample, o);
          }
        }
    });
  }
}

}  // namespace kernels
}  // namespace tiny_dnn
