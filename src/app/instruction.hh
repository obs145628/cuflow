#pragma once

#include <string>
#include <vector>
#include "tensor.hh"

struct Instruction
{
    std::string name;
    std::vector<std::string> args;
};

