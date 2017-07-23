/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/layers/cell.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

/*
 * optional parameters for the recurrent layer
*/
struct recurrent_layer_parameters {
  // the backend engine
  core::backend_t backend_type = core::default_engine();
  // whether to reset the state between timesteps
  bool reset_state = true;
};

/**
 * Wrapper for recurrent layers, manages the state of a recurrent cell.
 **/
class recurrent_layer : public layer {
 public:
  /**
   * @param cell [in] pointer to the wrapped cell
   * @param seq_len [in] length of the input sequences
   * @param params [in] recurrent layer optional parameters
   **/
  recurrent_layer(
    std::shared_ptr<cell> cell_p,
    size_t seq_len,
    const recurrent_layer_parameters params = recurrent_layer_parameters())
    : layer(cell_p->input_order(), cell_p->output_order()),
      cell_(cell_p),
      seq_len_(seq_len),
      reset_state_(params.reset_state) {
    layer::set_backend_type(params.backend_type);
    cell_->init_backend(
      static_cast<layer *>(this));  // depends on layer::set_backend_type!
  }

  // move constructor
  recurrent_layer(recurrent_layer &&other)
    : layer(std::move(other)),
      cell_(std::move(other.cell_)),
      seq_len_(std::move(other.seq_len_)) {
    cell_->init_backend(static_cast<layer *>(this));
  }

  size_t fan_in_size(size_t i) const override {
    return cell_->in_shape()[i].width_;
  }

  size_t fan_out_size(size_t i) const override {
    return cell_->in_shape()[i].height_;
  }

  std::vector<index3d<size_t>> in_shape() const override {
    return cell_->in_shape();
  }

  std::vector<index3d<size_t>> out_shape() const override {
    return cell_->out_shape();
  }

  void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) override {
    // TODO(prlz77): reshape input to sequences and handle states over time
    // (waiting for Xtensor)
    cell_->forward_propagation(in_data, out_data);
  }

  void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) override {
    // TODO(prlz77): reshape input to sequences and handle states over time
    // (waiting for Xtensor)
    cell_->back_propagation(in_data, out_data, out_grad, in_grad);
  }

  std::string layer_type() const override { return "recurrent-layer"; }

  friend struct serialization_buddy;

 private:
  // unique pointer to wrapped cell
  std::shared_ptr<cell> cell_;

  // sequence length
  size_t
    seq_len_;  // TODO(prlz77) std::vector, support variable sequence lengths

  bool reset_state_ = true;  // TODO(prlz77) reset state in truncated backprop
                             // when Xtensor implemented
};

}  // namespace tiny_dnn
