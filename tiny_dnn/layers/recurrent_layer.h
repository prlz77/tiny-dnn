/*
    Copyright (c) 2013, Taiga Nomi
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/layers/cell.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

/**
 * RNN layer.
 *
 * y(t-1)    y(t)   > h(t) = tanh(b + W*h(t-1) + U*x(t))
 *   ^        ^     > y(t) = c + V*h(t)
 *   |V+c     | V+c
 * h(t-1) -> h(t)
 *   ^ +b W   ^ +b
 *   |U       |U
 * x(t-1)    x(t)
 *
 **/
class recurrent_layer : public layer {
 public:
  /**
   * @param in_dim [in] number of elements of the input
   * @param out_dim [in] number of elements of the output
   * @param has_bias [in] whether to include additional bias to the layer
   * @param activation [in] activation function to be used internally
   **/
  recurrent_layer(cell&& cell_obj,
                       size_t seq_len)
    : layer(cell_obj.input_order(),
            cell_obj.output_order()),
            cell_(std::move(cell_obj)),
            seq_len_(seq_len) {
        layer::set_backend_type(cell_.get_backend_type());
        cell_.init_backend(layer::device());
  }

  // move constructor
  recurrent_layer(recurrent_layer &&other)
    : layer(std::move(other)),
      cell_(std::move(other.cell_)),
      seq_len_(std::move(other.seq_len_)) {
    cell_.init_backend(layer::device());
  }

  size_t fan_in_size(size_t i) const override {
    return cell_.in_shape()[i].width_;
  }

  size_t fan_out_size(size_t i) const override {
    return cell_.in_shape()[i].height_;
  }

  std::vector<index3d<size_t>> in_shape() const override {
    return cell_.in_shape();
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return cell_.out_shape();  // h(t)
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
      cell_.forward_propagation(in_data, out_data, layer::parallelize(), layer::engine());
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
      cell_.back_propagation(in_data, out_data, out_grad, in_grad, layer::parallelize(), layer::engine());
  }

  std::string layer_type() const override { return cell_.layer_type(); }

  friend struct serialization_buddy;

 private:
  cell&& cell_;

  size_t seq_len_;

};

}  // namespace tiny_dnn
