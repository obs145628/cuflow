#pragma once

#include <cstddef>

namespace gpu
{

    void op_matmul(const float* a, const float* b, float* out,
                   std::size_t arows, std::size_t acols, std::size_t bcols);

}
