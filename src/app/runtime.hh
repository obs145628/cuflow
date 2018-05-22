#pragma once

#include <map>
#include <string>
#include <vector>
#include <tocha/tensor.hh>
#include "instruction.hh"
#include "tensor.hh"


class Runtime
{

public:

    using ins_f = void (*)(Runtime& rt, const Instruction& ins);

    Runtime(const std::string& cmd_path,
            const std::string& in_path,
            const std::string& out_path);
    ~Runtime();
    Runtime(const Runtime&) = delete;
    Runtime& operator=(const Runtime&) = delete;

    void run();
    
    void run_ins(const std::vector<std::string>& args);

    void add_input(const std::string& name,
                   const std::vector<std::size_t>& shape);
    void add_output(const std::string& name,
                   const std::vector<std::size_t>& shape);

    Tensor* get_input(const std::string& name);
    Tensor* get_output(const std::string& name);
    Tensor* get_var(const std::string& name);

private:
    std::map<std::string, Tensor> inputs_;
    std::map<std::string, Tensor> outputs_;
    std::vector<std::string> inputs_list_;
    std::vector<std::string> outputs_list_;

    std::string cmd_path_;
    std::string out_path_;
    tocha::Tensors tin_;
    tocha::Tensors tout_;

    static std::map<std::string, ins_f> funs_map_;

public:
    static void add_fun(const std::string& name, ins_f f);
};
