/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include "tiny_dnn/core/params/lstm_cell_params.h"

namespace tiny_dnn {
namespace kernels {

inline void lstm_cell_op_internal(const tensor_t &x,
                                  const tensor_t &h_prev,
                                  const tensor_t &c_prev,
                                  const vec_t &W_x2i,
                                  const vec_t &W_x2f,
                                  const vec_t &W_x2c,
                                  const vec_t &W_x2o,
                                  const vec_t &W_h2i,
                                  const vec_t &W_h2f,
                                  const vec_t &W_h2c,
                                  const vec_t &W_h2o,
                                  const vec_t &b_2i,
                                  const vec_t &b_2f,
                                  const vec_t &b_2c,
                                  const vec_t &b_2o,
                                  tensor_t &out_data,
                                  tensor_t &h_next,
                                  tensor_t &c_next,
                                  tensor_t &pre_i,
                                  tensor_t &pre_f,
                                  tensor_t &pre_z,
                                  tensor_t &pre_o,
                                  tensor_t &post_i,
                                  tensor_t &post_f,
                                  tensor_t &post_z,
                                  tensor_t &post_c,
                                  const core::lstm_cell_params &params,
                                  const bool layer_parallelize) {
  for_(layer_parallelize, 0u, x.size(),
       [&](const blocked_range &r) {
         const size_t in_size  = params.in_size_;
         const size_t out_size = params.out_size_;
         auto tanh             = params.tanh_;
         auto sigmoid          = params.sigmoid_;
         const bool has_bias   = params.has_bias_;

         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           vec_t &pre_i_        = pre_i[sample];
           vec_t &pre_f_        = pre_f[sample];
           vec_t &pre_z_        = pre_z[sample];
           vec_t &pre_o_        = pre_o[sample];
           vec_t &post_i_       = post_i[sample];
           vec_t &post_f_       = post_f[sample];
           vec_t &post_z_       = post_z[sample];
           vec_t &post_c_       = post_c[sample];
           const vec_t &x_      = x[sample];
           const vec_t &h_prev_ = h_prev[sample];
           const vec_t &c_prev_ = c_prev[sample];
           vec_t &out_          = out_data[sample];
           vec_t &h_next_       = h_next[sample];
           vec_t &c_next_       = c_next[sample];
           for (size_t o = 0; o < out_size; o++) {
             float_t i_ = 0;
             float_t f_ = 0;
             float_t z_ = 0;
             float_t o_ = 0;
             for (size_t i = 0; i < in_size; i++) {
               i_ += W_x2i[i * out_size + o] * x_[i];
               f_ += W_x2f[i * out_size + o] * x_[i];
               z_ += W_x2c[i * out_size + o] * x_[i];
               o_ += W_x2o[i * out_size + o] * x_[i];
             }
             for (size_t o_2 = 0; o_2 < out_size; o_2++) {
               i_ += W_h2i[o_2 * out_size + o] * h_prev_[o];
               f_ += W_h2f[o_2 * out_size + o] * h_prev_[o];
               z_ += W_h2c[o_2 * out_size + o] * h_prev_[o];
               o_ += W_h2o[o_2 * out_size + o] * h_prev_[o];
             }
             if (has_bias) {
               i_ += b_2i[o];
               f_ += b_2f[o];
               z_ += b_2c[o];
               o_ += b_2o[o];
             }
             pre_i_[o] = i_;
             pre_f_[o] = f_;
             pre_z_[o] = z_;
             pre_o_[o] = o_;
           }

           sigmoid->forward_activation(pre_i_, post_i_);
           sigmoid->forward_activation(pre_f_, post_f_);
           sigmoid->forward_activation(pre_o_, out_);
           tanh->forward_activation(pre_z_, post_z_);

           for (size_t o = 0; o < out_size; o++) {
             c_next_[o] = post_f_[o] * c_prev_[o] + post_i_[o] * post_z_[o];
           }
           tanh->forward_activation(c_next_, post_c_);
           for (size_t o = 0; o < out_size; o++) {
             h_next_[o] = out_[o] * post_c_[o];
           }
         }
       },
       0u);  // for_i
}

inline void lstm_cell_op_internal(const tensor_t &x,
                                  const tensor_t &h_prev,
                                  const tensor_t &c_prev,
                                  const vec_t &W_x2i,
                                  const vec_t &W_x2f,
                                  const vec_t &W_x2c,
                                  const vec_t &W_x2o,
                                  const vec_t &W_h2i,
                                  const vec_t &W_h2f,
                                  const vec_t &W_h2c,
                                  const vec_t &W_h2o,
                                  tensor_t &dW_x2i,
                                  tensor_t &dW_x2f,
                                  tensor_t &dW_x2c,
                                  tensor_t &dW_x2o,
                                  tensor_t &dW_h2i,
                                  tensor_t &dW_h2f,
                                  tensor_t &dW_h2c,
                                  tensor_t &dW_h2o,
                                  tensor_t &db_2i,
                                  tensor_t &db_2f,
                                  tensor_t &db_2c,
                                  tensor_t &db_2o,
                                  const tensor_t d_o_next,
                                  const tensor_t d_h_next,
                                  const tensor_t d_c_next,
                                  tensor_t d_x_prev,
                                  tensor_t d_h_prev,
                                  tensor_t d_c_prev,
                                  const tensor_t o_next,
                                  const tensor_t c_next,
                                  const tensor_t pre_i,
                                  const tensor_t pre_f,
                                  const tensor_t pre_z,
                                  const tensor_t pre_o,
                                  const tensor_t post_i,
                                  const tensor_t post_f,
                                  const tensor_t post_z,
                                  const tensor_t post_c,
                                  const core::lstm_cell_params &params,
                                  const bool layer_parallelize) {
  for_(layer_parallelize, 0u, x.size(),
       [&](const blocked_range &r) {
         const size_t in_size  = params.in_size_;
         const size_t out_size = params.out_size_;
         auto tanh             = params.tanh_;
         auto sigmoid          = params.sigmoid_;
         const bool has_bias   = params.has_bias_;

         for (size_t sample = r.begin(); sample < r.end(); sample++) {
           const vec_t &d_h_next_ = d_h_next[sample];
           const vec_t &d_o_next_ = d_o_next[sample];
           const vec_t &c_next_   = c_next[sample];
           const vec_t &c_prev_   = c_prev[sample];
           const vec_t &o_next_   = o_next[sample];
           const vec_t &x_        = x[sample];
           const vec_t &h_prev_   = h_prev[sample];
           const vec_t &pre_i_    = pre_i[sample];
           const vec_t &pre_f_    = pre_f[sample];
           const vec_t &pre_z_    = pre_o[sample];
           const vec_t &pre_o_    = pre_o[sample];
           const vec_t &post_i_   = post_i[sample];
           const vec_t &post_c_   = post_c[sample];
           const vec_t &post_f_   = post_f[sample];
           const vec_t &post_z_   = post_z[sample];

           vec_t &dW_x2i_ = dW_x2i[sample];
           vec_t &dW_h2i_ = dW_h2i[sample];
           vec_t &dW_x2f_ = dW_x2f[sample];
           vec_t &dW_h2f_ = dW_h2f[sample];
           vec_t &dW_x2c_ = dW_x2c[sample];
           vec_t &dW_h2c_ = dW_h2c[sample];
           vec_t &dW_x2o_ = dW_x2o[sample];
           vec_t &dW_h2o_ = dW_h2o[sample];
           vec_t &db_2i_  = db_2i[sample];
           vec_t &db_2f_  = db_2f[sample];
           vec_t &db_2c_  = db_2c[sample];
           vec_t &db_2o_  = db_2o[sample];

           vec_t aux1(out_size);
           vec_t aux2(out_size);

           /** propagate deltas from o(t) to the inputs **/
           vec_t &d_x_prev_ = d_x_prev[sample];
           vec_t &d_h_prev_ = d_h_prev[sample];
           // aux1 = do(t) + dh(t) / do(t)
           for (size_t o = 0; o < out_size; o++) {
             aux1[o] =
               d_o_next_[o] + d_h_next_[o] * post_c_[o];  // dh - + -> do
           }
           sigmoid->backward_activation(pre_o_, o_next_, aux1, aux1);
           // dW2o
           for (size_t i = 0; i < in_size; i++) {
             for (size_t o = 0; o < out_size; o++) {
               dW_x2o_[i * out_size + o] = x_[i] * aux1[o];
             }
           }
           for (size_t o = 0; o < out_size; o++) {
             for (size_t o2 = 0; o2 < out_size; o2++) {
               dW_h2o_[o * out_size + o2] = h_prev_[o] * aux1[o2];
             }
           }
           if (has_bias) {
             for (size_t o = 0; o < out_size; o++) {
               db_2o_[o] = aux1[o];
             }
           }
           // dh->dx
           for (size_t i = 0; i < in_size; i++) {
             d_x_prev_[i] +=
               vectorize::dot(&W_x2o[i * out_size], &aux1[0], out_size);
           }
           // dh(t)->dh(t-1)
           for (size_t o = 0; o < out_size; o++) {
             d_h_prev_[o] +=
               vectorize::dot(&W_h2o[o * out_size], &aux1[0], out_size);
           }
           /** propagate deltas from h(t) to c(t) **/
           // aux1 = d/dc o(t)tanh(c(t))
           for (size_t o = 0; o < out_size; o++) {
             aux1[o] = d_h_next_[o] * o_next_[o];
           }
           // aux1 = dc(t) + dh(t) / dc(t)
           tanh->backward_activation(c_next_, post_c_, aux1, aux1);
           /** propagate deltas from c(t) to i(t) & inputs [x | h | c](t-1) **/
           const vec_t &d_c_next_ = d_c_next[sample];
           vec_t &d_c_prev_       = d_c_prev[sample];
           for (size_t o = 0; o < out_size; o++) {
             aux1[o] += d_c_next_[o];  // error coming from c(t)
             aux2[o]      = aux1[o] * post_z_[o];
             d_c_prev_[o] = aux1[o] * post_f_[o];  // dc(t-1)
           }
           // i to input; aux2 = di
           sigmoid->backward_activation(pre_i_, post_i_, aux2, aux2);
           // dW2i
           for (size_t i = 0; i < in_size; i++) {
             for (size_t o = 0; o < out_size; o++) {
               dW_x2i_[i * out_size + o] = x_[i] * aux2[o];
             }
           }
           for (size_t o = 0; o < out_size; o++) {
             for (size_t o2 = 0; o2 < out_size; o2++) {
               dW_h2i_[o * out_size + o2] = h_prev_[o2] * aux2[o2];
             }
           }
           if (has_bias) {
             for (size_t o = 0; o < out_size; o++) {
               db_2i_[o] = aux2[o];
             }
           }
           // di->dx input
           for (size_t i = 0; i < in_size; i++) {
             d_x_prev_[i] +=
               vectorize::dot(&W_x2i[i * out_size], &aux2[0], out_size);
           }
           // di->dh(t-1)
           for (size_t o = 0; o < out_size; o++) {
             d_h_prev_[o] +=
               vectorize::dot(&W_h2i[o * out_size], &aux2[0], out_size);
           }
           // aux2 can be reused from here
           /** propagate deltas from c(t) to z & inputs **/
           for (size_t o = 0; o < out_size; o++) {
             aux2[o] = aux1[o] * post_i_[o];
           }
           // z to input; aux2 = dz
           tanh->backward_activation(pre_z_, post_z_, aux2, aux2);
           // dW2c
           for (size_t i = 0; i < in_size; i++) {
             for (size_t o = 0; o < out_size; o++) {
               dW_x2c_[i * out_size + o] = x_[i] * aux2[o];
             }
           }
           for (size_t o = 0; o < out_size; o++) {
             for (size_t o2 = 0; o2 < out_size; o2++) {
               dW_h2c_[o * out_size + o2] = h_prev_[o2] * aux2[o2];
             }
           }
           if (has_bias) {
             for (size_t o = 0; o < out_size; o++) {
               db_2c_[o] = aux2[o];
             }
           }
           // dz->dx input
           for (size_t i = 0; i < in_size; i++) {
             d_x_prev_[i] +=
               vectorize::dot(&W_x2c[i * out_size], &aux2[0], out_size);
           }
           // di->dh(t-1)
           for (size_t o = 0; o < out_size; o++) {
             d_h_prev_[o] +=
               vectorize::dot(&W_h2c[o * out_size], &aux2[0], out_size);
           }
           // Note:: an aux3 can be used to merge these two blocks to reduce
           // the "for" overhead at the expense of memory.
           // aux2 can be reused from here
           /** propagate deltas from f(t) to inputs **/
           for (size_t o = 0; o < out_size; o++) {
             aux2[o] = aux1[o] * c_prev_[o];
           }
           sigmoid->backward_activation(pre_f_, post_f_, aux2, aux2);
           // dW2f
           for (size_t i = 0; i < in_size; i++) {
             for (size_t o = 0; o < out_size; o++) {
               dW_x2f_[i * out_size + o] = x_[i] * aux2[o];
             }
           }
           for (size_t o = 0; o < out_size; o++) {
             for (size_t o2 = 0; o2 < out_size; o2++) {
               dW_h2f_[o * out_size + o2] = h_prev_[o2] * aux2[o2];
             }
           }
           if (has_bias) {
             for (size_t o = 0; o < out_size; o++) {
               db_2f_[o] = aux2[o];
             }
           }
           // d -> input
           for (size_t i = 0; i < in_size; i++) {
             d_x_prev_[i] +=
               vectorize::dot(&W_x2f[i * out_size], &aux2[0], out_size);
           }
           for (size_t o = 0; o < out_size; o++) {
             d_h_prev_[o] +=
               vectorize::dot(&W_h2f[o * out_size], &aux2[0], out_size);
           }
         }

       },
       0u);  // for_i
}
}  // namespace kernels
}  // namespace tiny_dnn
