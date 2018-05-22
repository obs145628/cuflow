#include "runtime.hh"
#include <cassert>
#include <stdexcept>
#include <fstream>
#include <sstream>
#include "instruction.hh"


Runtime::Runtime(const std::string& cmd_path,
                 const std::string& in_path,
                 const std::string& out_path)
    : cmd_path_(cmd_path)
    , out_path_(out_path)
    , tin_(tocha::Tensors::load(in_path))
{
    
}


Runtime::~Runtime()
{
    for (auto it : inputs_)
    {
        cudaFree(it.second.gdata);
    }
    for (auto it : outputs_)
    {
        cudaFree(it.second.gdata);
    }
}


float* test = nullptr;


void Runtime::run()
{
    std::ifstream cmds(cmd_path_);
    std::string line;

    while (std::getline(cmds, line))
    {
        std::istringstream iline(line);
        std::string s;
        std::vector<std::string> args;
        while(std::getline(iline, s, ','))
            args.push_back(s);
        run_ins(args);
    }

    for (std::size_t i = 0; i < outputs_list_.size(); ++i)
    {
        auto t = get_output(outputs_list_[i]);
        assert(t);

        auto& tdata = tout_.arr()[i];
        cudaMemcpy(t->cdata, t->gdata, t->size * sizeof(float), cudaMemcpyDeviceToHost);        
    }

    tout_.save(out_path_);
}

void Runtime::run_ins(const std::vector<std::string>& args)
{
    Instruction ins;
    ins.name = args[0];
    ins.args = args;

    auto it = funs_map_.find(ins.name);
    if (it == funs_map_.end())
        throw std::runtime_error {"Can't find instruction: " + ins.name};
    it->second(*this, ins);
}

void Runtime::add_input(const std::string& name,
                        const std::vector<std::size_t>& shape)
{
    auto& tdata = tin_.arr()[inputs_list_.size()];
    float* cdata = reinterpret_cast<float*>(tdata.data);
    float* gdata;
    cudaMalloc(&gdata, tdata.total_len * sizeof(float));
    cudaMemcpy(gdata, cdata, tdata.total_len * sizeof(float), cudaMemcpyHostToDevice);

    test = gdata;

    Tensor t(shape, cdata, gdata);
    inputs_.insert({name, t});
    inputs_list_.push_back(name);
}

void Runtime::add_output(const std::string& name,
                         const std::vector<std::size_t>& shape)
{
    
    if (shape.size() == 0)
        tout_.add(tocha::Tensor::f32());
    else if (shape.size() == 1)
        tout_.add(tocha::Tensor::f32(shape[0]));
    else if (shape.size() == 2)
        tout_.add(tocha::Tensor::f32(shape[0], shape[1]));
    else if (shape.size() == 3)
        tout_.add(tocha::Tensor::f32(shape[0], shape[1], shape[2]));
    else if (shape.size() == 4)
        tout_.add(tocha::Tensor::f32(shape[0], shape[1], shape[2], shape[3]));
    
    auto& tdata = tout_.arr()[outputs_list_.size()];

    float* cdata = reinterpret_cast<float*>(tdata.data);
    float* gdata;
    cudaMalloc(&gdata, tdata.total_len * sizeof(float));

    Tensor t(shape, cdata, gdata);
    outputs_.insert({name, t});
    outputs_list_.push_back(name);
}


Tensor* Runtime::get_input(const std::string& name)
{
    auto it = inputs_.find(name);
    return it == inputs_.end() ? nullptr : &it->second;
}

Tensor* Runtime::get_output(const std::string& name)
{
    auto it = outputs_.find(name);
    return it == inputs_.end() ? nullptr : &it->second;
}

Tensor* Runtime::get_var(const std::string& name)
{
    auto res = get_input(name);
    if (res)
        return res;

    res = get_output(name);
    if (res)
        return res;

    return nullptr;
}




std::map<std::string, Runtime::ins_f> Runtime::funs_map_ {};

void Runtime::add_fun(const std::string& name, ins_f f)
{
    funs_map_[name] = f;
}
