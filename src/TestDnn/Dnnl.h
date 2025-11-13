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

#include "Types.h"
#include "Tensor.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#if defined(__linux__)
#include "oneapi/dnnl/dnnl.hpp"

namespace td
{
    inline void Copy(const Tensor & src, dnnl::memory& dst) 
    {
        if(dst.get_engine().get_kind() != dnnl::engine::kind::cpu)
            throw std::runtime_error("Copy supports only CPU memory!");
        if (src.RawSize() != dst.get_desc().get_size())
            throw std::runtime_error("Copy: input and output have different sizes!");
        if(dst.get_data_handle() == nullptr)
            throw std::runtime_error("Copy: check output!");
        memcpy(dst.get_data_handle(), src.RawData(), src.RawSize());
    }

    inline void Copy(const dnnl::memory& src, Tensor& dst)
    {
        if (src.get_engine().get_kind() != dnnl::engine::kind::cpu)
            throw std::runtime_error("Copy supports only CPU memory!");
        if (dst.RawSize() != src.get_desc().get_size())
            throw std::runtime_error("Copy: input and output have different sizes!");
        if (src.get_data_handle() == nullptr)
            throw std::runtime_error("Copy: check input!");
        memcpy(dst.RawData(), src.get_data_handle(), src.get_desc().get_size());
    }

    inline void ToBf16(const Tensor& src, dnnl::memory& dst)
    {
        SimdFloat32ToBFloat16(src.Data<float>(), src.Size(), (uint16_t*)dst.get_data_handle());
    }

    inline void ToFp32(const dnnl::memory& src, Tensor& dst)
    {
        SimdBFloat16ToFloat32((uint16_t*)src.get_data_handle(), dst.Size(), dst.Data<float>());
    }
}
#endif