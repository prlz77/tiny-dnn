/*
    Copyright (c) 2013, Taiga Nomi and the respective contributors
    All rights reserved.

    Use of this source code is governed by a BSD-style license that can be found
    in the LICENSE file.
*/
#pragma once

#include <gtest/gtest.h>

#include <functional>
#include <vector>

#include "test/testhelper.h"
#include "tiny_dnn/tiny_dnn.h"

using namespace tiny_dnn::activation;

namespace tiny_dnn {

TEST(gru, train) {
  network<sequential> nn;
  adagrad optimizer;
  recurrent_layer_parameters params;
  params.reset_state = false;
  nn << recurrent_layer(gru(3, 2), 1, params);
  nn.weight_init(weight_init::xavier());

  vec_t a(3), t(2), a2(3), t2(2);

  // clang-format off
    a[0] = 3.0; a[1] = 0.0; a[2] = -1.0;
    t[0] = 0.3; t[1] = 0.7;

    a2[0] = 0.2; a2[1] = 0.5; a2[2] = 4.0;
    t2[0] = 0.5; t2[1] = 0.1;
  // clang-format on

  std::vector<vec_t> data, train;

  for (size_t i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 0.1;
  nn.train<mse>(optimizer, data, train, 1, 20);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-5);
  EXPECT_NEAR(predicted[1], t[1], 1E-5);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-5);
  EXPECT_NEAR(predicted[1], t2[1], 1E-5);
}

TEST(gru, train_different_batches) {
  auto batch_sizes = {2, 7, 10, 12};
  size_t data_size = std::accumulate(batch_sizes.begin(), batch_sizes.end(), 1,
                                     std::multiplies<size_t>());
  for (auto &batch_sz : batch_sizes) {
    network<sequential> nn;
    adagrad optimizer;

    nn << recurrent_layer(gru(3, 2), 1) << fully_connected_layer(2, 2);
    nn.weight_init(weight_init::xavier());

    vec_t a(3), t(2), a2(3), t2(2);

    // clang-format off
a[0] = 3.0; a[1] = 0.0; a[2] = -1.0;
t[0] = 0.3; t[1] = 0.7;

a2[0] = 0.2; a2[1] = 0.5; a2[2] = 4.0;
t2[0] = 0.5; t2[1] = 0.1;
    // clang-format on

    std::vector<vec_t> data, train;

    for (size_t i = 0; i < data_size; i++) {
      data.push_back(a);
      data.push_back(a2);
      train.push_back(t);
      train.push_back(t2);
    }
    optimizer.alpha = 0.1;
    nn.train<mse>(optimizer, data, train, batch_sz, 10);

    vec_t predicted = nn.predict(a);

    EXPECT_NEAR(predicted[0], t[0], 1E-5);
    EXPECT_NEAR(predicted[1], t[1], 1E-5);

    predicted = nn.predict(a2);

    EXPECT_NEAR(predicted[0], t2[0], 1E-5);
    EXPECT_NEAR(predicted[1], t2[1], 1E-5);
  }
}

TEST(gru, train2) {
  network<sequential> nn;
  gradient_descent optimizer;

  nn << recurrent_layer(gru(4, 6), 1) << recurrent_layer(gru(6, 3), 1);
  nn.weight_init(weight_init::xavier());

  vec_t a(4, 0.0), t(3, 0.0), a2(4, 0.0), t2(3, 0.0);

  // clang-format off
a[0] = 3.0; a[1] = 1.0; a[2] = -1.0; a[3] = 4.0;
t[0] = 0.3; t[1] = 0.7; t[2] = 0.3;

a2[0] = 1.0; a2[1] = 0.0; a2[2] = 4.0; a2[3] = 2.0;
t2[0] = 0.6; t2[1] = 0.1; t2[2] = 0.1; // 0.0 is difficult due to sigmoid.
  // clang-format on

  std::vector<vec_t> data, train;

  for (size_t i = 0; i < 100; i++) {
    data.push_back(a);
    data.push_back(a2);
    train.push_back(t);
    train.push_back(t2);
  }
  optimizer.alpha = 2.0;
  nn.train<mse>(optimizer, data, train, 1, 20);

  vec_t predicted = nn.predict(a);

  EXPECT_NEAR(predicted[0], t[0], 1E-4);
  EXPECT_NEAR(predicted[1], t[1], 1E-4);

  predicted = nn.predict(a2);

  EXPECT_NEAR(predicted[0], t2[0], 1E-4);
  EXPECT_NEAR(predicted[1], t2[1], 1E-4);
}

TEST(gru, gradient_check) {
  network<sequential> nn;
  nn << recurrent_layer(gru(50, 10), 1);

  const auto test_data = generate_gradient_check_data(nn.in_data_size());
  nn.init_weight();
  EXPECT_TRUE(nn.gradient_check<mse>(test_data.first, test_data.second,
                                     epsilon<float_t>(), GRAD_CHECK_ALL));
}

TEST(gru, read_write) {
  recurrent_layer l1(gru(100, 100), 1);
  recurrent_layer l2(gru(100, 100), 1);

  l1.setup(true);
  l2.setup(true);

  serialization_test(l1, l2);
}

}  // namespace tiny-dnn
