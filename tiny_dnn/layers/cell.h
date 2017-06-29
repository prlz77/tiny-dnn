/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once
#include "tiny_dnn/core/backend.h"
#include "tiny_dnn/layers/layer.h"

namespace tiny_dnn {

/**
 * compute fully-connected(matmul) operation
 **/
class cell {
 public:
  virtual std::vector<vector_type> input_order() = 0;

  virtual std::vector<vector_type> output_order() = 0;

  virtual size_t fan_in_size(size_t i) const = 0;

  virtual size_t fan_out_size(size_t i) const = 0;

  virtual std::vector<index3d<size_t>> in_shape() const = 0;

  virtual std::vector<index3d<size_t>> out_shape() const = 0;

  virtual void forward_propagation(const std::vector<tensor_t *> &in_data,
                           std::vector<tensor_t *> &out_data) = 0;

  virtual void back_propagation(const std::vector<tensor_t *> &in_data,
                        const std::vector<tensor_t *> &out_data,
                        std::vector<tensor_t *> &out_grad,
                        std::vector<tensor_t *> &in_grad) = 0;

  virtual std::string layer_type() const = 0;

  virtual void init_backend(layer* layer_p) = 0;

};

}  // namespace tiny_dnn
