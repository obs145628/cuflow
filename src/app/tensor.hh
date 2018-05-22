#pragma once

#include <vector>


struct Tensor
{
    const std::vector<std::size_t> shape;
    const std::size_t size;
    float* cdata;
    float* gdata;

    static std::size_t tensor_size(const std::vector<std::size_t>& shape)
    {
        std::size_t res = 1;
        for (auto x : shape)
            res *= x;
        return res;
    }
    

    Tensor(const Tensor& t)
        : shape(t.shape)
        , size(t.size)
        , cdata(t.cdata)
        , gdata(t.gdata)
        {}

    Tensor(const std::vector<std::size_t>& shape,
           float* cdata, float* gdata)
        : shape(shape)
        , size(tensor_size(shape))
        , cdata(cdata)
        , gdata(gdata)
        {}
};
