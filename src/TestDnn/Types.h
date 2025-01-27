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

#include "Simd/SimdAllocator.hpp"

#include <vector>
#include <memory.h>

namespace TestDnn
{
    typedef Cpl::String String;
    typedef Cpl::Strings Strings;

    typedef std::vector<size_t> Shape;
    typedef std::vector<size_t> Index;

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

    class Tensor
    {
    public:
        typedef SimdTensorDataType Type;
        typedef SimdTensorFormatType Format;
        typedef std::vector<uint8_t, Simd::Allocator<uint8_t>> Buffer;

        SIMD_INLINE Tensor()
            : _type(SimdTensorDataUnknown)
            , _format(SimdTensorFormatUnknown)
            , _size(0)
        {
        }

        SIMD_INLINE Tensor(Type type, const Shape& shape, Format format = SimdTensorFormatUnknown)
            : _type(type)
            , _shape(shape)
            , _size(0)
        {
            Resize();
        }

        template<class U> SIMD_INLINE Tensor(Type type, const Shape& shape, const Format& format, const U& value)
            : _type(type)
            , _shape(shape)
            , _format(format)
        {
            Resize<U>(value);
        }

        SIMD_INLINE ~Tensor()
        {
        }

        SIMD_INLINE void Reshape(Type type, const Shape& shape, const Format& format = SimdTensorFormatUnknown)
        {
            _type = type;
            _shape = shape;
            _format = format;
            Resize();
        }

        template<class U> SIMD_INLINE void Reshape(Type type, const Shape& shape, const Format& format, const U& value)
        {
            _type = type;
            _shape = shape;
            _format = format;
            Resize<U>(value);
        }

        SIMD_INLINE void Extend(Type type, const Shape& shape, const Format& format = SimdTensorFormatUnknown)
        {
            if (_type == SimdTensorDataUnknown)
                _type = type;
            else
                assert(_type == type);
            assert(_type != SimdTensorDataUnknown);
            _shape = shape;
            _format = format;
            _size = Size(0, _shape.size());
            size_t size = _size * TypeSize(_type);
            if (size > _buffer.size())
                _buffer.resize(size);
        }

        SIMD_INLINE void Clone(const Tensor& tensor)
        {
            _shape = tensor._shape;
            _format = tensor._format;
            _size = tensor._size;
            _buffer = tensor._buffer;
        }

        SIMD_INLINE Type GetType()
        {
            return _type;
        }

        SIMD_INLINE Format GetFormat() const
        {
            return _format;
        }

        SIMD_INLINE const Shape& GetShape() const
        {
            return _shape;
        }

        SIMD_INLINE size_t Count() const
        {
            return _shape.size();
        }

        SIMD_INLINE size_t Index(ptrdiff_t axis) const
        {
            if (axis < 0)
                axis += _shape.size();
            return axis;
        }

        SIMD_INLINE size_t Axis(ptrdiff_t axis) const
        {
            return _shape[Index(axis)];
        }

        SIMD_INLINE size_t Size(ptrdiff_t startAxis, ptrdiff_t endAxis) const
        {
            startAxis = Index(startAxis);
            endAxis = Index(endAxis);
            assert(startAxis <= endAxis && (size_t)endAxis <= _shape.size());
            size_t size = 1;
            for (ptrdiff_t axis = startAxis; axis < endAxis; ++axis)
                size *= _shape[axis];
            return size;
        }

        SIMD_INLINE size_t Size(ptrdiff_t startAxis) const
        {
            return Size(startAxis, _shape.size());
        }

        SIMD_INLINE size_t Size() const
        {
            return _size;
        }

        SIMD_INLINE size_t Offset(const TestDnn::Index& index) const
        {
            assert(_shape.size() == index.size());

            size_t offset = 0;
            for (size_t axis = 0; axis < _shape.size(); ++axis)
            {
                assert(_shape[axis] > 0);
                assert(index[axis] < _shape[axis]);

                offset *= _shape[axis];
                offset += index[axis];
            }
            return offset;
        }

        SIMD_INLINE size_t RawSize() const
        {
            return _buffer.size();
        }

        SIMD_INLINE uint8_t* RawData()
        {
            return _buffer.data();
        }

        SIMD_INLINE const uint8_t* RawData() const
        {
            return _buffer.data();
        }

        template<class U> SIMD_INLINE U* Data()
        {
            return (U*)_buffer.data();
        }

        template<class U> SIMD_INLINE const U* Data() const
        {
            assert(_type == GetTensorType<U>() || _buffer->data == NULL);
            return (U*)_buffer.data();
        }

        template<class U> SIMD_INLINE U* Data(const TestDnn::Index& index)
        {
            return Data<U>() + Offset(index);
        }

        template<class U> SIMD_INLINE const U* Data(const TestDnn::Index& index) const
        {
            return Data<U>() + Offset(index);
        }

        static SIMD_INLINE size_t TypeSize(Type type)
        {
            switch (type)
            {
            case SimdTensorDataUnknown: return 0;
            case SimdTensorData32f: return 4;
            case SimdTensorData32i: return 4;
            case SimdTensorData8i: return 1;
            case SimdTensorData8u: return 1;
            case SimdTensorData64i: return 8;
            case SimdTensorData64u: return 8;
            case SimdTensorDataBool: return 1;
            case SimdTensorData16b: return 2;
            case SimdTensorData16f: return 2;
            default: assert(0); return 0;
            }
        }

        template<class U> SIMD_INLINE void Resize(U value)
        {
            _size = Size(0, _shape.size());
            _buffer.resize(_size * TypeSize(_type));
            for (size_t i = 0; i < _size; ++i)
                ((U*)_buffer.data())[i] = value;
        }

        SIMD_INLINE void Resize()
        {
            assert(_type != SimdTensorDataUnknown);
            _size = Size(0, _shape.size());
            _buffer.resize(_size * TypeSize(_type));
            memset(_buffer.data(), 0, _buffer.size());
        }

    private:
        Type _type;
        Format _format;
        Shape _shape;
        size_t _size;
        Buffer _buffer;
    };
}