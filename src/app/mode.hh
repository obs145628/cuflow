#pragma once

#include <fstream>

static constexpr int MODE_GPU = 1;
static constexpr int MODE_CPU = 2;

extern int arch_mode;

extern std::ofstream logs;
