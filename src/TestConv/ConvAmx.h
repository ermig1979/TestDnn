#pragma once

#include <cstddef>
#include <cstdint>

namespace tcv
{
    struct Param
    {
        int batch;
        int srcH, srcW, srcC;
        int dstC;
        int kernelY, kernelX;
        int strideY, strideX;
        int dilationY, dilationX;
        int padY, padX, padH, padW;
        bool relu;
    };

    class ConvAmx
    {
    public:
        ConvAmx();
        ~ConvAmx();

        bool Init(const Param& param, const float* weight, const float* bias);
        void SetSrc(const uint16_t* src);
        void Run(uint16_t* dst);

    private:
        struct Buffer;

        Param _p;
        Buffer* _buf;
        const uint16_t* _src;
    };
}
