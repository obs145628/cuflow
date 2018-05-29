cmake_minimum_required(VERSION 3.8 FATAL_ERROR)
project(cuflow LANGUAGES CXX CUDA)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS}")

include_directories(ext/tocha/include)
find_package(ZLIB REQUIRED)

set(CUFLOW_SRC
  src/app/init.cu
  src/app/init.hh
  src/app/instruction.cu
  src/app/instruction.hh
  src/app/main.cu
  src/app/memory.hh
  src/app/memory.hxx
  src/app/mode.hh
  src/app/mode.cu
  src/app/runtime.cu
  src/app/runtime.hh
  src/app/tensor.hh

  src/ops_cpu/vadd.hh
  src/ops_cpu/vadd.cu

  src/ops_gpu/vadd.hh
  src/ops_gpu/vadd.cu
)
add_executable(cuflow ${CUFLOW_SRC})
target_link_libraries(cuflow ${ZLIB_LIBRARIES})



add_custom_target(check
   COMMAND
   python ${CMAKE_SOURCE_DIR}/tests/main.py
   ${CMAKE_SOURCE_DIR}
   ${CMAKE_BINARY_DIR}
   DEPENDS cuflow
)
