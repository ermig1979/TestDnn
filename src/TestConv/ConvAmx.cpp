#include "ConvAmx.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace tcv
{
    namespace
    {
        struct alignas(64) TileConf
        {
            uint8_t paletteId;
            uint8_t startRow;
            uint8_t reserved[14];
            uint16_t colsb[16];
            uint8_t rows[16];
        };

        void ConfigureTiles(int rowsA, bool twoM)
        {
            TileConf conf;
            std::memset(&conf, 0, sizeof(conf));
            conf.paletteId = 1;
            for (int i = 0; i < 8; ++i)
                conf.colsb[i] = 64;
            for (int i = 0; i < 8; ++i)
                conf.rows[i] = 16;
            if (twoM)
            {
                if (rowsA != 16)
                {
                    conf.rows[0] = conf.rows[1] = conf.rows[2] = conf.rows[3] = (uint8_t)rowsA;
                    conf.rows[4] = conf.rows[5] = (uint8_t)rowsA;
                }
            }
            else
            {
                conf.rows[0] = (uint8_t)rowsA;
                conf.rows[1] = (uint8_t)rowsA;
                conf.rows[4] = (uint8_t)rowsA;
            }
            _tile_loadconfig(&conf);
        }

        uint16_t FloatToBf16(float value)
        {
            uint32_t bits;
            std::memcpy(&bits, &value, sizeof(bits));
            uint32_t lsb = (bits >> 16) & 1u;
            bits += 0x7fffu + lsb;
            return (uint16_t)(bits >> 16);
        }

        void StoreActivation(const float* acc, int rows, int oc0, const float* bias, uint16_t* dst, int dstC, bool relu)
        {
            const __m512 zero = _mm512_setzero_ps();
            for (int r = 0; r < rows; ++r)
            {
                const float* row = acc + r * 32;
                uint16_t* out = dst + r * dstC + oc0;
                for (int c = 0; c < 32; c += 16)
                {
                    __m512 v = _mm512_add_ps(_mm512_loadu_ps(row + c), _mm512_loadu_ps(bias + oc0 + c));
                    if (relu)
                        v = _mm512_max_ps(v, zero);
                    __m256bh packed = _mm512_cvtneps_pbh(v);
                    _mm256_storeu_si256((__m256i*)(out + c), (__m256i)packed);
                }
            }
        }

        // Two output rows, 16 (or fewer) columns each. The halves are not contiguous,
        // so each K step loads them from their own pointer and shares the weight tile.
        void GemmRows(const uint16_t* src, int oh0, int oh1, int ow, int wp, int ic, int kh, int kw, const uint16_t* weight, float* acc)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            const int strideS = ic * (int)sizeof(uint16_t);
            const int nK = kh * kw * (ic / 32);
            const uint16_t* b = weight;
            const uint16_t* a1 = src + ((size_t)oh1 * wp + ow) * ic;
            _tile_loadd(4, src + ((size_t)oh0 * wp + ow) * ic, strideS);
            _tile_stream_loadd(6, b, 64);
            int step = 0;
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    const uint16_t* row0 = src + ((size_t)(oh0 + y) * wp + (ow + x)) * ic;
                    const uint16_t* row1 = src + ((size_t)(oh1 + y) * wp + (ow + x)) * ic;
                    for (int c = 0; c < ic; c += 32, ++step)
                    {
                        if (step == 0)
                            continue;
                        const uint16_t* next1 = row1 + c;
                        const uint16_t* nextB = weight + (size_t)step * 1024;
                        _tile_loadd(5, a1, strideS);
                        _tile_stream_loadd(7, b + 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(4, row0 + c, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_stream_loadd(6, nextB, 64);
                        _tile_dpbf16ps(3, 5, 7);
                        a1 = next1;
                        b = nextB;
                    }
                }
            }
            if (nK > 0)
            {
                _tile_loadd(5, a1, strideS);
                _tile_stream_loadd(7, b + 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stored(0, acc, 128);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stored(1, acc + 16, 128);
                _tile_dpbf16ps(2, 5, 6);
                _tile_stored(2, acc + 16 * 32, 128);
                _tile_dpbf16ps(3, 5, 7);
                _tile_stored(3, acc + 16 * 32 + 16, 128);
            }
        }

        void Gemm16x32(const uint16_t* src, int oh, int ow, int wp, int ic, int kh, int kw, const uint16_t* weight, float* acc)
        {
            _tile_zero(0);
            _tile_zero(1);
            const int strideS = ic * (int)sizeof(uint16_t);
            const uint16_t* b = weight;
            const int nK = kh * kw * (ic / 32);
            _tile_loadd(4, src + (oh * wp + ow) * ic, strideS);
            _tile_stream_loadd(6, b, 64);
            int step = 0;
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    const uint16_t* row = src + ((oh + y) * wp + (ow + x)) * ic;
                    for (int c = 0; c < ic; c += 32, ++step)
                    {
                        const uint16_t* nextA = row + c;
                        const uint16_t* nextB = weight + (size_t)step * 1024;
                        if (step == 0)
                            continue;
                        _tile_stream_loadd(7, b + 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(4, nextA, strideS);
                        _tile_stream_loadd(6, nextB, 64);
                        b = nextB;
                    }
                }
            }
            if (nK > 0)
            {
                _tile_stream_loadd(7, b + 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stored(0, acc, 128);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stored(1, acc + 16, 128);
            }
        }

        void Gemm32x32(const uint16_t* src, int oh, int ow, int wp, int ic, int kh, int kw, const uint16_t* weight, float* acc)
        {
            _tile_zero(0);
            _tile_zero(1);
            _tile_zero(2);
            _tile_zero(3);
            const int strideS = ic * (int)sizeof(uint16_t);
            const int nK = kh * kw * (ic / 32);
            const uint16_t* b = weight;
            const uint16_t* a0 = src + (oh * wp + ow) * ic;
            _tile_loadd(4, a0, strideS);
            _tile_stream_loadd(6, b, 64);
            int step = 0;
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    const uint16_t* row = src + ((oh + y) * wp + (ow + x)) * ic;
                    for (int c = 0; c < ic; c += 32, ++step)
                    {
                        if (step == 0)
                            continue;
                        const uint16_t* nextA = row + c;
                        const uint16_t* nextB = weight + (size_t)step * 1024;
                        _tile_loadd(5, a0 + 16 * ic, strideS);
                        _tile_stream_loadd(7, b + 512, 64);
                        _tile_dpbf16ps(0, 4, 6);
                        _tile_dpbf16ps(1, 4, 7);
                        _tile_loadd(4, nextA, strideS);
                        _tile_dpbf16ps(2, 5, 6);
                        _tile_stream_loadd(6, nextB, 64);
                        _tile_dpbf16ps(3, 5, 7);
                        a0 = nextA;
                        b = nextB;
                    }
                }
            }
            if (nK > 0)
            {
                _tile_loadd(5, a0 + 16 * ic, strideS);
                _tile_stream_loadd(7, b + 512, 64);
                _tile_dpbf16ps(0, 4, 6);
                _tile_stored(0, acc, 128);
                _tile_dpbf16ps(1, 4, 7);
                _tile_stored(1, acc + 16, 128);
                _tile_dpbf16ps(2, 5, 6);
                _tile_stored(2, acc + 16 * 32, 128);
                _tile_dpbf16ps(3, 5, 7);
                _tile_stored(3, acc + 16 * 32 + 16, 128);
            }
        }
    }

    struct AlignedU16
    {
        uint16_t* data = nullptr;
        size_t count = 0;

        ~AlignedU16()
        {
            std::free(data);
        }

        void assign(size_t n, uint16_t value)
        {
            if (count != n)
            {
                std::free(data);
                void* ptr = nullptr;
                if (posix_memalign(&ptr, 64, n * sizeof(uint16_t)) != 0)
                    ptr = nullptr;
                data = (uint16_t*)ptr;
                count = data ? n : 0;
            }
            if (data)
                std::fill(data, data + count, value);
        }
    };

    struct ConvAmx::Buffer
    {
        AlignedU16 weight;
        std::vector<float> bias;
        AlignedU16 src;
        int wp = 0;
        int hp = 0;
        int dstH = 0;
        int dstW = 0;
    };

    ConvAmx::ConvAmx()
        : _buf(new Buffer)
        , _src(nullptr)
    {
    }

    ConvAmx::~ConvAmx()
    {
        delete _buf;
    }

    bool ConvAmx::Init(const Param& param, const float* weight, const float* bias)
    {
        if (param.batch < 1 || param.strideY != 1 || param.strideX != 1 || param.dilationY != 1 || param.dilationX != 1)
            return false;
        if (param.srcC < 32 || param.dstC < 32 || (param.srcC % 32) != 0 || (param.dstC % 32) != 0)
            return false;
        if (param.kernelY < 1 || param.kernelX < 1)
            return false;
        _p = param;
        _buf->hp = param.srcH + param.padY + param.padH;
        _buf->wp = param.srcW + param.padX + param.padW;
        _buf->dstH = (param.srcH + param.padY + param.padH - param.kernelY) / param.strideY + 1;
        _buf->dstW = (param.srcW + param.padX + param.padW - param.kernelX) / param.strideX + 1;
        _buf->src.assign((size_t)_buf->hp * _buf->wp * param.srcC, 0);
        _buf->bias.assign(bias, bias + param.dstC);

        const int ic = param.srcC;
        const int oc = param.dstC;
        const int kh = param.kernelY;
        const int kw = param.kernelX;
        _buf->weight.assign((size_t)(oc / 32) * kh * kw * (ic / 32) * 1024, 0);
        uint16_t* dst = _buf->weight.data;
        for (int oc0 = 0; oc0 < oc; oc0 += 32)
        {
            for (int y = 0; y < kh; ++y)
            {
                for (int x = 0; x < kw; ++x)
                {
                    for (int ic0 = 0; ic0 < ic; ic0 += 32)
                    {
                        for (int tile = 0; tile < 2; ++tile)
                        {
                            for (int pair = 0; pair < 16; ++pair)
                            {
                                for (int n = 0; n < 16; ++n)
                                {
                                    int c0 = ic0 + pair * 2;
                                    int o = oc0 + tile * 16 + n;
                                    const float* w = weight + ((y * kw + x) * ic + c0) * oc + o;
                                    *dst++ = FloatToBf16(w[0]);
                                    *dst++ = FloatToBf16(w[oc]);
                                }
                            }
                        }
                    }
                }
            }
        }
        return true;
    }

    void ConvAmx::SetSrc(const uint16_t* src)
    {
        _src = src;
    }

    void ConvAmx::Run(uint16_t* dst)
    {
        const Param& p = _p;
        const int ic = p.srcC;
        const int oc = p.dstC;
        const int kh = p.kernelY;
        const int kw = p.kernelX;
        const int wp = _buf->wp;
        const int srcH = p.srcH;
        const int srcW = p.srcW;
        const int dstH = _buf->dstH;
        const int dstW = _buf->dstW;
        uint16_t* srcPad = _buf->src.data;
        std::memset(srcPad, 0, _buf->src.count * sizeof(uint16_t));
        for (int n = 0; n < p.batch; ++n)
        {
            const uint16_t* srcN = _src + (size_t)n * srcH * srcW * ic;
            uint16_t* dstN = dst + (size_t)n * dstH * dstW * oc;
            for (int y = 0; y < srcH; ++y)
            {
                const uint16_t* row = srcN + (size_t)y * srcW * ic;
                uint16_t* dstRow = srcPad + ((y + p.padY) * wp + p.padX) * ic;
                std::memcpy(dstRow, row, (size_t)srcW * ic * sizeof(uint16_t));
            }

            alignas(64) float acc[32 * 32];
            const size_t weightBlock = (size_t)kh * kw * (ic / 32) * 1024;
            if (dstW < 32)
            {
                const size_t panelBytes = weightBlock * sizeof(uint16_t);
                const size_t budget = 1536u * 1024u;
                const int pairs = dstH / 2;
                int chunkPairs = pairs > 0 ? pairs : 1;
                const size_t imageBytes = (size_t)_buf->hp * wp * ic * sizeof(uint16_t);
                if (pairs > 1 && imageBytes + panelBytes > budget)
                {
                    chunkPairs = 1;
                    for (int np = 1; np <= pairs; ++np)
                    {
                        int h = std::min(_buf->hp, 2 * np + kh - 1);
                        size_t bytes = (size_t)h * wp * ic * sizeof(uint16_t) + panelBytes;
                        if (bytes <= budget)
                            chunkPairs = np;
                    }
                }
                const int full = dstW / 16;
                const int rem = dstW % 16;
                for (int pc = 0; pc < pairs; pc += chunkPairs)
                {
                    int np = std::min(chunkPairs, pairs - pc);
                    if (full > 0)
                    {
                        ConfigureTiles(16, true);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* weight = _buf->weight.data + (size_t)(oc0 / 32) * weightBlock;
                            for (int pi = 0; pi < np; ++pi)
                            {
                                int oh0 = (pc + pi) * 2;
                                int oh1 = oh0 + 1;
                                for (int b = 0; b < full; ++b)
                                {
                                    int ow = b * 16;
                                    GemmRows(srcPad, oh0, oh1, ow, wp, ic, kh, kw, weight, acc);
                                    StoreActivation(acc, 16, oc0, _buf->bias.data(), dstN + ((size_t)oh0 * dstW + ow) * oc, oc, p.relu);
                                    StoreActivation(acc + 16 * 32, 16, oc0, _buf->bias.data(), dstN + ((size_t)oh1 * dstW + ow) * oc, oc, p.relu);
                                }
                            }
                        }
                    }
                    if (rem > 0)
                    {
                        ConfigureTiles(rem, true);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* weight = _buf->weight.data + (size_t)(oc0 / 32) * weightBlock;
                            for (int pi = 0; pi < np; ++pi)
                            {
                                int oh0 = (pc + pi) * 2;
                                int oh1 = oh0 + 1;
                                int ow = full * 16;
                                GemmRows(srcPad, oh0, oh1, ow, wp, ic, kh, kw, weight, acc);
                                StoreActivation(acc, rem, oc0, _buf->bias.data(), dstN + ((size_t)oh0 * dstW + ow) * oc, oc, p.relu);
                                StoreActivation(acc + 16 * 32, rem, oc0, _buf->bias.data(), dstN + ((size_t)oh1 * dstW + ow) * oc, oc, p.relu);
                            }
                        }
                    }
                }
                if ((dstH % 2) != 0)
                {
                    int oh = dstH - 1;
                    int ow = 0;
                    for (; ow + 16 <= dstW; ow += 16)
                    {
                        ConfigureTiles(16, false);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* weight = _buf->weight.data + (size_t)(oc0 / 32) * weightBlock;
                            Gemm16x32(srcPad, oh, ow, wp, ic, kh, kw, weight, acc);
                            StoreActivation(acc, 16, oc0, _buf->bias.data(), dstN + ((size_t)oh * dstW + ow) * oc, oc, p.relu);
                        }
                    }
                    if (ow < dstW)
                    {
                        int rows = dstW - ow;
                        ConfigureTiles(rows, false);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* weight = _buf->weight.data + (size_t)(oc0 / 32) * weightBlock;
                            Gemm16x32(srcPad, oh, ow, wp, ic, kh, kw, weight, acc);
                            StoreActivation(acc, rows, oc0, _buf->bias.data(), dstN + ((size_t)oh * dstW + ow) * oc, oc, p.relu);
                        }
                    }
                }
            }
            else
            {
                ConfigureTiles(16, true);
                for (int oh = 0; oh < dstH; ++oh)
                {
                    int ow = 0;
                    for (; ow + 32 <= dstW; ow += 32)
                    {
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* weight = _buf->weight.data + (size_t)(oc0 / 32) * weightBlock;
                            Gemm32x32(srcPad, oh, ow, wp, ic, kh, kw, weight, acc);
                            StoreActivation(acc, 32, oc0, _buf->bias.data(), dstN + (oh * dstW + ow) * oc, oc, p.relu);
                        }
                    }
                    while (ow < dstW)
                    {
                        int rows = std::min(16, dstW - ow);
                        ConfigureTiles(rows, false);
                        for (int oc0 = 0; oc0 < oc; oc0 += 32)
                        {
                            const uint16_t* weight = _buf->weight.data + (size_t)(oc0 / 32) * weightBlock;
                            Gemm16x32(srcPad, oh, ow, wp, ic, kh, kw, weight, acc);
                            StoreActivation(acc, rows, oc0, _buf->bias.data(), dstN + (oh * dstW + ow) * oc, oc, p.relu);
                        }
                        ow += rows;
                        ConfigureTiles(16, true);
                    }
                }
            }
        }
        _tile_release();
    }
}
