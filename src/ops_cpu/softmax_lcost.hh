#pragma once

#include <cstddef>

namespace cpu
{

    void op_softmax_lcost(const float* y, const float* logits, float* out,
                          std::size_t m, std::size_t n);

}
