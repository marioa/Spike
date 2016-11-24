#pragma once

#include "Spike/STDP/STDP.hpp"
#include "Spike/Backend/CUDA/CUDABackend.hpp"
#include <cuda.h>
#include <vector_types.h>
#include <curand.h>
#include <curand_kernel.h>

namespace Backend {
  namespace CUDA {
    class STDPCommon : public virtual ::Backend::STDPCommon {
    public:
    };

    class STDP : public virtual ::Backend::CUDA::STDPCommon,
                 public ::Backend::STDP {
    public:
    };
  }
}