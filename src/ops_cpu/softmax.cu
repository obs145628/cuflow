#include "matmul.hh"
#include "../app/mode.hh"
#include "../app/timer.hh"
#include <cmath>

namespace cpu
{

    namespace
    {

        float max(const float* begin, const float* end)
        {
            float res = *begin;
            for (const float* it = begin; it != end; ++it)
                res = std::max(res, *it);
            return res;
        }

        float sum(const float* begin, const float* end)
        {
            float res = 0;
            for (const float* it = begin; it != end; ++it)
                res += *it;
            return res;
        }

    }
    

    void op_softmax(const float* x, float* y,
                    std::size_t m, std::size_t n)
    {
        auto start = timer::now();
        

        for (std::size_t i = 0; i < m; ++i)
        {

            //max_x = max(x[i])
            float max_x = max(x + i * n, x + (i + 1) * n);

            //y[i] = exp(x[i] - max_x)
            for (std::size_t j = 0; j < n; ++j)
                y[i * n + j] = std::exp(x[i * n + j] - max_x);

            
            //sum_ex = sum(y[i])
            float sum_ex = sum(y + i * n, y + (i + 1) * n);


            //y[i] = y[i] / sum_ex
            for (std::size_t j = 0; j < n; ++j)
                y[i * n + j] = y[i * n + j] / sum_ex;
        }


        auto time = timer::now() - start;
        logs << "[CPU_SOFTMAX]: " << time << "ms.\n";
    }

}
