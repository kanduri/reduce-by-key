#pragma once

#ifdef __HIPCC__
#include <hip/hip_runtime.h>
#define GPU(expr) hip##expr
#elif defined(__CUDACC__)
#include <cuda_runtime_api.h>
#include "cuda_common.hpp"
#define GPU(expr) cuda##expr
#endif

inline void gpu_check(GPU(Error_t) error, const std::string &file, int line) {
  if (error != GPU(Success)) {
    std::cerr << "GPU error: " << GPU(GetErrorName(error)) << " at " << file
              << ":" << line << std::endl;
    std::abort();
  }
}

#define GPU_CALL(expr) gpu_check(GPU(expr), __FILE__, __LINE__)

#if __CUDA_ARCH__ < 600 // Maxwell or older (no native double precision atomic addition)
    __device__
    inline double GPU_atomic_add(double* address, double val) {
        using I = unsigned long long int;
        I* address_as_ull = (I*)address;
        I old = *address_as_ull, assumed;
        do {
            assumed = old;
            old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val+__longlong_as_double(assumed)));
        } while (assumed != old);
        return __longlong_as_double(old);
    }
#else // use build in atomicAdd for double precision from Pascal onwards
    __device__
    inline double GPU_atomic_add(double* address, double val) {
        return atomicAdd(address, val);
    }
#endif

__device__
inline double GPU_atomic_sub(double* address, double val) {
    return GPU_atomic_add(address, -val);
}

__device__
inline float GPU_atomic_add(float* address, float val) {
    return atomicAdd(address, val);
}

__device__
inline float GPU_atomic_sub(float* address, float val) {
    return atomicAdd(address, -val);
}

