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

#include <vector>
#include <memory.h>

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
}