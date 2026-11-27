/*
* Test DNN Project (http://github.com/ermig1979/TestDnn).
*
* Copyright (c) 2025-2025 Yermalayeu Ihar.
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in
* all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#pragma once 

#include "Cpl/Log.h"
#include "Cpl/Args.h"
#include "Cpl/Performance.h"

#include "Simd/SimdAllocator.hpp"
#include "Simd/SimdPoint.hpp"
#include "Simd/SimdLib.hpp"

#include <vector>
#include <cstring>
#include <memory.h>

#if defined(__linux__)
#include <unistd.h>
#include <sys/syscall.h>
#endif

namespace td
{
    typedef Cpl::String String;
    typedef Cpl::Strings Strings;

    typedef std::vector<size_t> Shape;
    typedef std::vector<size_t> Index;

    typedef Simd::Point<size_t> Size;

    typedef std::vector<int64_t> Dims;

    //--------------------------------------------------------------------------------------------------

    CPL_INLINE Shape Shp()
    {
        return Shape();
    }

    CPL_INLINE Shape Shp(size_t axis0)
    {
        return Shape({ axis0 });
    }

    CPL_INLINE Shape Shp(size_t axis0, size_t axis1)
    {
        return Shape({ axis0, axis1 });
    }

    CPL_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2)
    {
        return Shape({ axis0, axis1, axis2 });
    }

    CPL_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2, size_t axis3)
    {
        return Shape({ axis0, axis1, axis2, axis3 });
    }

    CPL_INLINE Shape Shp(size_t axis0, size_t axis1, size_t axis2, size_t axis3, size_t axis4)
    {
        return Shape({ axis0, axis1, axis2, axis3, axis4 });
    }

    //--------------------------------------------------------------------------------------------------

    CPL_INLINE Dims Dms()
    {
        return Dims();
    }

    CPL_INLINE Dims Dms(int64_t axis0)
    {
        return Dims({ axis0 });
    }

    CPL_INLINE Dims Dms(int64_t axis0, int64_t axis1)
    {
        return Dims({ axis0, axis1 });
    }

    CPL_INLINE Dims Dms(int64_t axis0, int64_t axis1, int64_t axis2)
    {
        return Dims({ axis0, axis1, axis2 });
    }

    CPL_INLINE Dims Dms(int64_t axis0, int64_t axis1, int64_t axis2, int64_t axis3)
    {
        return Dims({ axis0, axis1, axis2, axis3 });
    }

    CPL_INLINE Dims Dms(int64_t axis0, int64_t axis1, int64_t axis2, int64_t axis3, int64_t axis4)
    {
        return Dims({ axis0, axis1, axis2, axis3, axis4 });
    }

    //--------------------------------------------------------------------------------------------------

    CPL_INLINE int Rand()
    {
        return ::rand();
    }

    CPL_INLINE void Srand(unsigned int seed)
    {
        ::srand(seed);
    }

    CPL_INLINE float Random()
    {
        return float(Rand() & INT16_MAX) / float(INT16_MAX);
    }

    //--------------------------------------------------------------------------------------------------

    static bool PinThread(size_t core)
    {
#if defined(__linux__)
        pthread_t this_thread = pthread_self();
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(core, &cpuset);
        if (pthread_setaffinity_np(this_thread, sizeof(cpu_set_t), &cpuset))
        {
            CPL_LOG_SS(Warning, "Can't set affinity " << core << " to " << this_thread << " thread : " << std::strerror(errno) << " !");
            return false;
        }
#if 0
        int policy = 0;
        sched_param params;
        if (pthread_getschedparam(this_thread, &policy, &params))
        {
            CPL_LOG_SS(Warning, "Can't get thread parameters of " << this_thread << " thread : " << std::strerror(errno) << " !");
            return false;
        }

        CPL_LOG_SS(Info, "Core : " << core << " Policy: " << policy << " Priority: " << params.sched_priority << " max: " << sched_get_priority_max(policy) << " min: " << sched_get_priority_min(policy));

        if (pthread_setschedparam(this_thread, policy, &params))
        {
            CPL_LOG_SS(Warning, "Can't set thread parameters of " << this_thread << " thread : " << std::strerror(errno) << " !");
            return false;
        }

        CPL_LOG_SS(Info, "Core : " << core << " Policy: " << policy << " Priority: " << params.sched_priority);
#endif
#endif
        return true;
    }

    static String CoreFreqInfo()
    {
        std::stringstream info;
#if defined(__linux__)
        info << " " << sched_getcpu();
#if defined(TD_SIMD)
        info << ": " << Cpl::ToStr(double(SimdCpuInfo(SimdCpuInfoCurrentFrequency)) / 1000000000.0, 1) << " GHz.";
#endif
#endif
        return info.str();
    }

    inline void Float32ToBFloat16(const float* src, size_t size, uint16_t* dst)
    {
#if defined(TD_SIMD)
        ::SimdFloat32ToBFloat16(src, size, dst);
#else
        for (size_t i = 0; i < size; ++i)
        {
            uint32_t bits;
            std::memcpy(&bits, src + i, sizeof(bits));
            uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7fffu + lsb;
            dst[i] = (uint16_t)(bits >> 16);
        }
#endif
    }

    inline void BFloat16ToFloat32(const uint16_t* src, size_t size, float* dst)
    {
#if defined(TD_SIMD)
        ::SimdBFloat16ToFloat32(src, size, dst);
#else
        for (size_t i = 0; i < size; ++i)
        {
            uint32_t bits = (uint32_t)src[i] << 16;
            std::memcpy(dst + i, &bits, sizeof(bits));
        }
#endif
    }

    inline void EnableAmx()
    {
#if defined(__linux__)
        const long ARCH_REQ_XCOMP_PERM = 0x1023;
        const long XFEATURE_XTILEDATA = 18;
        if (syscall(SYS_arch_prctl, ARCH_REQ_XCOMP_PERM, XFEATURE_XTILEDATA) != 0)
            CPL_LOG_SS(Error, "AMX tile data permission was not granted.");
#endif
    }

    inline void LitterCpuCache(size_t k)
    {
#if defined(TD_SIMD)
        Simd::LitterCpuCache(k);
#else
        (void)k;
#endif
    }
}