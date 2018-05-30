#include "matmul.hh"
#include "../app/mode.hh"
#include "../app/timer.hh"

namespace cpu
{

    void op_matmul(const float* a, const float* b, float* out,
                   std::size_t arows, std::size_t acols, std::size_t bcols)
    {
        auto start = timer::now();
        
        
        for (std::size_t i = 0; i < arows; ++i)
            for (std::size_t j = 0; j < bcols; ++j)
            {
                float x = 0;
                for (std::size_t k = 0; k < acols; ++k)
                    x += a[i * acols + k] * b[k * bcols + j];
                out[i * bcols + j] = x;
            }


        auto time = timer::now() - start;
        logs << "[CPU_MATMUL]: " << time << "ms.\n";
    }

}
