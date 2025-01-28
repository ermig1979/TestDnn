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

#include "Tensor.h"
#include "ConvParam.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

//#include "example_utils.hpp"
#include "oneapi/dnnl/dnnl.hpp"

namespace TestDnn
{
	class TestConvolution32f
	{
	public:
		virtual ~TestConvolution32f() {};
		virtual String Name() const = 0;
		virtual bool Init(const ConvParam & param, const Tensor& weigth, const Tensor& bias, const Tensor& params) = 0;
		virtual bool SetSrc(const Tensor& src) = 0;
		virtual bool Run() = 0;
		virtual bool GetDst(Tensor& dst) = 0;
		TestConvolution32f& Ref() { return *this; }
	};

	//----------------------------------------------------------------------------------------------------

	class TestConvolution32fSimd : public TestConvolution32f
	{
		void* _context;
		Tensor _buf, _src, _dst;
	public:
		TestConvolution32fSimd()
			: _context(nullptr)
		{
		}

		virtual ~TestConvolution32fSimd()
		{
			if (_context)
			{
				SimdRelease(_context);
				_context = nullptr;
			}
		}

		virtual String Name() const
		{
			return "Simd";
		}

		virtual bool Init(const ConvParam& param, const Tensor& weight, const Tensor& bias, const Tensor& params)
		{
			_context = SimdSynetConvolution32fInit(param.batch, &param.conv);
			if (!_context)
				return false;

			_buf.Reshape(SimdTensorData32f, Shp(SimdSynetConvolution32fExternalBufferSize(_context)));
			
			SimdSynetConvolution32fSetParams(_context, weight.Data<float>(), NULL, bias.Data<float>(), params.Data<float>());

			return true;
		}

		virtual bool SetSrc(const Tensor& src)
		{
			_src.Clone(src);
			return true;
		}

		virtual bool Run()
		{
			if(_context)
				SimdSynetConvolution32fForward(_context, _src.Data<float>(), _buf.Data<float>(), _dst.Data<float>());
			return true;
		}

		virtual bool GetDst(Tensor& dst)
		{
			dst.Clone(_dst);
			return true;
		}
	};

	//----------------------------------------------------------------------------------------------------

	bool Convolution32fTest(float eps, const ConvParam& p, TestConvolution32f &f1, TestConvolution32f &f2)
	{
		CPL_LOG_SS(Info, "Test " << f1.Name() << " & " << f2.Name() << " for " << p.Description() << ": ");

		const SimdConvolutionParameters& c = p.conv;
		Tensor src(c.srcT, Shp(p.batch, p.trans ? c.srcH : c.srcC, p.trans ? c.srcW : c.srcH, p.trans ? c.srcC : c.srcW));
		Random32f(src);

		Tensor weight(c.srcT, Shp(p.trans ? c.kernelY : c.dstC, p.trans ? c.kernelX : c.srcC / c.group,
			p.trans ? c.srcC / c.group : c.kernelY, p.trans ? c.dstC : c.kernelX));
		Random32f(weight);

		Tensor bias(c.srcT, Shp(c.dstC));
		Random32f(bias);

		Tensor params(c.srcT, Shp(c.dstC));
		Random32f(params);

		if (c.activation == ::SimdConvolutionActivationHswish)
		{
			params.Data<float>()[0] = 3.0f;
			params.Data<float>()[1] = 1.0f / 6.0f;
		}
		else if (c.activation == ::SimdConvolutionActivationMish)
			params.Data<float>()[0] = 20.0f;
		else if (c.activation == ::SimdConvolutionActivationHardSigmoid)
		{
			params.Data<float>()[0] = 1.0f / 6.0f;
			params.Data<float>()[1] = 0.5f;
		}
		else
		{
			params.Data<float>()[0] = 0.1f;
			params.Data<float>()[1] = 1.1f;
		}

		Tensor dst1(c.dstT, Shp(p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW));
		Tensor dst2(c.dstT, Shp(p.batch, p.trans ? c.dstH : c.dstC, p.trans ? c.dstW : c.dstH, p.trans ? c.dstC : c.dstW));

		if (!f1.Init(p, weight, bias, params))
			return false;
		if (!f2.Init(p, weight, bias, params))
			return false;

		f1.SetSrc(src);
		f2.SetSrc(src);

		{
			CPL_PERF_BEGF(f1.Name() + p.Description(), p.Flop());
			//f1.Run();
		}

		f1.GetDst(dst1);
		f2.GetDst(dst2);

		return Compare32f(dst1, dst2, eps, true, 64);
	}

	bool Convolution32fTest()
	{
		Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _6(6, 6), _7(7, 7);
		const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
			aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
			aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
			aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
		const SimdBool tF = SimdFalse, tT = SimdTrue;
		const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
		float eps = 0.001f;

		bool result = true;

		Cpl::PerformanceStorage::Global().Clear();

		result = result && Convolution32fTest(eps, ConvParam(1, 128, 12, 12, 128, _1, _1, _1, _0, _0, 1, aRe, tT), TestConvolution32fSimd().Ref(), TestConvolution32fSimd().Ref());
		result = result && Convolution32fTest(eps, ConvParam(1, 384, 13, 13, 1152, _1, _1, _1, _0, _0, 1, aRe, tT), TestConvolution32fSimd().Ref(), TestConvolution32fSimd().Ref());

		CPL_LOG_SS(Info, std::endl << Cpl::PerformanceStorage::Global().Report());

		return result;
	}
}