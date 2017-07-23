/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/gru_cell_params.h"

namespace tiny_dnn {
namespace kernels {

inline void gru_cell_op_internal(const tensor_t &x,
                                 const tensor_t &h_prev,
                                 const vec_t &W_x2z,
                                 const vec_t &W_x2r,
                                 const vec_t &W_x2h,
                                 const vec_t &W_hr2c,
                                 const vec_t &W_s2z,
                                 const vec_t &W_s2r,
                                 const vec_t &b_2z,
                                 const vec_t &b_2r,
                                 const vec_t &b_2h,
                                 tensor_t &out,
                                 tensor_t &post_h,
                                 tensor_t &post_r,
                                 tensor_t &post_z,
                                 tensor_t &pre_h,
                                 tensor_t &pre_r,
                                 tensor_t &hr,
                                 tensor_t &pre_z,
                                 tensor_t &z_neg,
                                 const core::gru_cell_params &params,
                                 const bool layer_parallelize) {
  for_(layer_parallelize, 0u, x.size(),
       [&](const blocked_range &r) {
         const size_t in_size  = params.in_size_;
         const size_t out_size = params.out_size_;
         auto tanh             = params.tanh_;
         auto sigmoid          = params.sigmoid_;
         const bool has_bias   = params.has_bias_;

         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec_t &x_      = x[sample];
           const vec_t &h_prev_ = h_prev[sample];
           vec_t &out_          = out[sample];
           vec_t &post_h_       = post_h[sample];
           vec_t &post_r_       = post_r[sample];
           vec_t &post_z_       = post_z[sample];
           vec_t &pre_h_        = pre_h[sample];
           vec_t &hr_           = hr[sample];
           vec_t &pre_r_        = pre_r[sample];
           vec_t &pre_z_        = pre_z[sample];
           vec_t &z_neg_        = z_neg[sample];

           for (size_t o = 0; o < out_size; o++) {
             float_t local_z = 0.0;
             float_t local_r = 0.0;
             float_t local_h = 0.0;
             for (size_t i = 0; i < in_size; i++) {
               local_z += W_x2z[i * out_size + o] * x_[i];
               local_r += W_x2r[i * out_size + o] * x_[i];
               local_h += W_x2h[i * out_size + o] * x_[i];
             }
             for (size_t o_2 = 0; o_2 < out_size; o_2++) {
               local_z += W_s2z[o_2 * out_size + o] * h_prev_[o_2];
               local_r += W_s2r[o_2 * out_size + o] * h_prev_[o_2];
             }
             if (has_bias) {
               local_z += b_2z[o];
               local_r += b_2r[o];
               local_h += b_2h[o];
             }
             pre_z_[o] = local_z;
             pre_r_[o] = local_r;
             pre_h_[o] = local_h;
           }
           sigmoid->forward_activation(pre_z_, post_z_);
           sigmoid->forward_activation(pre_r_, post_r_);

           for (size_t o = 0; o < out_size; o++) {  // from
             out_[o]    = h_prev_[o] * post_z_[o];
             z_neg_[o]  = 1 - post_z_[o];
             float_t hr = h_prev_[o] * post_r_[o];
             for (size_t o_2 = 0; o_2 < out_size; o_2++) {  // to
               pre_h_[o_2] += W_hr2c[o * out_size + o_2] * hr;
             }
             hr_[o] = hr;
           }
           tanh->forward_activation(pre_h_, post_h_);
           for (size_t o = 0; o < out_size; o++) {
             out_[o] += z_neg_[o] * post_h_[o];
           }
         }
       },
       0u);  // for_i
}

inline void gru_cell_op_internal(const tensor_t &x,
                                 const tensor_t &h_prev,
                                 vec_t &W_x2z,
                                 vec_t &W_x2r,
                                 vec_t &W_x2h,
                                 vec_t &W_hr2c,
                                 vec_t &W_s2z,
                                 vec_t &W_s2r,
                                 tensor_t &dW_x2z,
                                 tensor_t &dW_x2r,
                                 tensor_t &dW_x2h,
                                 tensor_t &dW_hr2c,
                                 tensor_t &dW_s2z,
                                 tensor_t &dW_s2r,
                                 tensor_t &db_2z,
                                 tensor_t &db_2r,
                                 tensor_t &db_2h,
                                 const tensor_t d_o_next,
                                 tensor_t d_x_prev,
                                 tensor_t d_h_prev,
                                 const tensor_t post_h,
                                 const tensor_t post_r,
                                 const tensor_t post_z,
                                 const tensor_t pre_h,
                                 const tensor_t hr,
                                 const tensor_t pre_r,
                                 const tensor_t pre_z,
                                 const tensor_t z_neg,
                                 const core::gru_cell_params &params,
                                 const bool layer_parallelize) {
  for_(
    layer_parallelize, 0u, x.size(),
    [&](const blocked_range &r) {
      const size_t in_size  = params.in_size_;
      const size_t out_size = params.out_size_;
      auto tanh             = params.tanh_;
      auto sigmoid          = params.sigmoid_;
      const bool has_bias   = params.has_bias_;

      for (size_t sample = r.begin(); sample < r.end(); sample++) {
        const vec_t x_         = x[sample];
        const vec_t h_prev_    = h_prev[sample];
        vec_t &dW_x2z_         = dW_x2z[sample];
        vec_t &dW_x2r_         = dW_x2r[sample];
        vec_t &dW_x2h_         = dW_x2h[sample];
        vec_t &dW_hr2c_        = dW_hr2c[sample];
        vec_t &dW_s2z_         = dW_s2z[sample];
        vec_t &dW_s2r_         = dW_s2r[sample];
        vec_t &db_2z_          = db_2z[sample];
        vec_t &db_2r_          = db_2r[sample];
        vec_t &db_2h_          = db_2h[sample];
        const vec_t &d_o_next_ = d_o_next[sample];
        vec_t &d_x_prev_       = d_x_prev[sample];
        vec_t &d_h_prev_       = d_h_prev[sample];
        const vec_t &post_h_   = post_h[sample];
        const vec_t &post_r_   = post_r[sample];
        const vec_t &post_z_   = post_z[sample];
        const vec_t &pre_h_    = pre_h[sample];
        const vec_t &hr_       = hr[sample];
        const vec_t &pre_r_    = pre_r[sample];
        const vec_t &pre_z_    = pre_z[sample];
        const vec_t &z_neg_    = z_neg[sample];

        vec_t aux1(out_size);

        // do -> ds(t-1), do -> dz
        for (size_t o = 0; o < out_size; o++) {
          d_h_prev_[o] += d_o_next_[o] * post_z_[o];
          aux1[o] = d_o_next_[o] * (h_prev_[o] - post_h_[o]);  // aux1 = dz
        }
        sigmoid->backward_activation(pre_z_, post_z_, aux1, aux1);

        // dz -> db2z
        if (has_bias) {
          for (size_t o = 0; o < out_size; o++) {
            db_2z_[o] = aux1[o];
          }
        }
        // from dz -> x
        for (size_t i = 0; i < in_size; i++) {
          d_x_prev_[i] +=
            vectorize::dot(&aux1[0], &W_x2z[i * out_size], out_size);
        }
        // dW_x2z
        for (size_t i = 0; i < in_size; i++) {
          vectorize::muladd(&aux1[0], x_[i], out_size, &dW_x2z_[i * out_size]);
        }
        // from dz -> h_prev
        for (size_t o = 0; o < out_size; o++) {
          d_h_prev_[o] +=
            vectorize::dot(&aux1[0], &W_s2z[o * out_size], out_size);
        }
        // dW_s2z
        for (size_t o = 0; o < out_size; o++) {
          vectorize::muladd(&aux1[0], h_prev_[o], out_size,
                            &dW_s2z_[o * out_size]);
        }
        // do->dh_next
        for (size_t o = 0; o < out_size; o++) {
          aux1[o] = d_o_next_[o] * z_neg_[o];
        }

        tanh->backward_activation(pre_h_, post_h_, aux1, aux1);

        // dh -> db2h
        if (has_bias) {
          for (size_t o = 0; o < out_size; o++) {
            db_2h_[o] = aux1[o];
          }
        }

        // dh_prev -> dx
        for (size_t i = 0; i < in_size; i++) {
          d_x_prev_[i] +=
            vectorize::dot(&aux1[0], &W_x2h[i * out_size], out_size);
        }
        // dh_prev -> dWx2h
        for (size_t i = 0; i < in_size; i++) {
          vectorize::muladd(&aux1[0], x_[i], out_size, &dW_x2h_[i * out_size]);
        }
        // dh_prev -> dWhr2c
        for (size_t o = 0; o < out_size; o++) {
          vectorize::muladd(&aux1[0], hr_[o], out_size,
                            &dW_hr2c_[o * out_size]);
        }
        // dWhr -> dhr
        for (size_t o = 0; o < out_size; o++) {
          aux1[o] = vectorize::dot(&aux1[0], &W_hr2c[o * out_size], out_size);
        }
        // dhr -> dh_prev
        for (size_t o = 0; o < out_size; o++) {
          d_h_prev_[o] = aux1[o] * post_r_[o];
        }
        // dhr -> dr
        for (size_t o = 0; o < out_size; o++) {
          aux1[o] = aux1[o] * h_prev_[o];
        }

        sigmoid->backward_activation(pre_r_, post_r_, aux1, aux1);

        // dr -> db2r
        if (has_bias) {
          for (size_t o = 0; o < out_size; o++) {
            db_2r_[o] = aux1[o];
          }
        }

        // dr -> dx
        for (size_t i = 0; i < in_size; i++) {
          d_x_prev_[i] +=
            vectorize::dot(&aux1[0], &W_x2r[i * out_size], out_size);
        }
        // dr -> dWx2r
        for (size_t i = 0; i < in_size; i++) {
          vectorize::muladd(&aux1[0], x_[i], out_size, &dW_x2r_[out_size * i]);
        }
        // dr -> dh_prev
        for (size_t o = 0; o < out_size; o++) {
          d_h_prev_[o] +=
            vectorize::dot(&aux1[0], &W_s2r[o * out_size], out_size);
        }
        // dr -> dWs2r
        for (size_t o = 0; o < out_size; o++) {
          vectorize::muladd(&aux1[0], h_prev_[o], out_size,
                            &dW_s2r_[out_size * o]);
        }
      }
    },
    0u);  // for_i
}
}  // namespace kernels
}  // namespace tiny_dnn
