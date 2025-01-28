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
#include "Options.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>

#include "../examples/example_utils.hpp"
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

			SimdSynetConvolution32fSetParams(_context, weight.Data<float>(), NULL, bias.Data<float>(), params.Data<float>());

			_src.Reshape(SimdTensorData32f, param.SrcShape());

			_buf.Reshape(SimdTensorData32f, Shp(SimdSynetConvolution32fExternalBufferSize(_context)));

			_dst.Reshape(SimdTensorData32f, param.DstShape());

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

	class TestConvolution32fDnnl : public TestConvolution32f
	{
		using tag = dnnl::memory::format_tag;
		using dt = dnnl::memory::data_type;

		dnnl::engine _engine;
		dnnl::stream _engine_stream;

		dnnl::convolution_forward::primitive_desc _convPd;
		dnnl::convolution_forward _convPrim;
		std::unordered_map<int, dnnl::memory> _convArgs;

		dnnl::memory::format_tag _formatS, _formatW;
		dnnl::memory::dims _srcDims, _dstDims, _weightDims, _biasDims;

		dnnl::memory _userSrcMem, _userWeightMem, _userBiasMem, _userDstMem;
		dnnl::memory::desc _srcMd, _weightMd, _userBiasMd, _dstMd;
		dnnl::memory _convSrcMem, _convWeightMem, _convDstMem;

		dnnl::memory::dims ToDims(const Shape& shape) const
		{
			dnnl::memory::dims dims(shape.size());
			for (size_t i = 0; i < shape.size(); ++i)
				dims[i] = shape[i];
			return dims;
		}

	public:
		TestConvolution32fDnnl()
			: _engine(dnnl::engine::kind::cpu, 0)
			, _engine_stream(_engine)
		{

		}

		virtual ~TestConvolution32fDnnl()
		{

		}

		virtual String Name() const
		{
			return "Dnnl";
		}

		virtual bool Init(const ConvParam& p, const Tensor& weight, const Tensor& bias, const Tensor& params)
		{
			const SimdConvolutionParameters& c = p.conv;

			_formatS = c.srcF == SimdTensorFormatNhwc ? tag::nhwc : tag::nchw;
			_formatW = c.srcF == SimdTensorFormatNhwc ? tag::hwio : tag::oihw;

			_srcDims = ToDims(Shp(p.batch, c.srcC, c.srcH, c.srcW));
			_weightDims = ToDims(Shp(c.dstC, c.srcC, c.kernelY, c.kernelX));
			_biasDims = ToDims(Shp(c.dstC));
			_dstDims = ToDims(Shp(p.batch, c.dstC, c.dstH, c.dstW));

			_userSrcMem = dnnl::memory({ _srcDims, dt::f32, _formatS }, _engine);
			_userWeightMem = dnnl::memory({ _weightDims, dt::f32, _formatW }, _engine);
			_userDstMem = dnnl::memory({ _dstDims, dt::f32, _formatS }, _engine);

			_srcMd = dnnl::memory::desc(_srcDims, dt::f32, tag::any);
			_weightMd = dnnl::memory::desc(_weightDims, dt::f32, tag::any);
			_dstMd = dnnl::memory::desc(_dstDims, dt::f32, tag::any);

			_userBiasMd = dnnl::memory::desc(_biasDims, dt::f32, tag::a);
			_userBiasMem = dnnl::memory(_userBiasMd, _engine);

			write_to_dnnl_memory((void*)weight.RawData(), _userWeightMem);
			write_to_dnnl_memory((void*)bias.RawData(), _userBiasMem);

			// Create primitive post-ops (ReLU).
			const float alpha = 0.f;
			const float beta = 0.f;
			dnnl::post_ops conv_ops;
			conv_ops.append_eltwise(dnnl::algorithm::eltwise_relu, alpha, beta);
			dnnl::primitive_attr conv_attr;
			conv_attr.set_post_ops(conv_ops);
			conv_attr.set_fpmath_mode(dnnl::fpmath_mode::bf16);

			dnnl::memory::dims strides_dims = { c.strideY, c.strideX };
			dnnl::memory::dims padding_dims_l = { c.padY, c.padX };
			dnnl::memory::dims padding_dims_r = { c.padH, c.padW };

			_convPd = dnnl::convolution_forward::primitive_desc(_engine,
				dnnl::prop_kind::forward_inference, dnnl::algorithm::convolution_direct,
				_srcMd, _weightMd, _userBiasMd, _dstMd,
				strides_dims, padding_dims_l, padding_dims_r, conv_attr);

			_convSrcMem = _userSrcMem;
			if (_convPd.src_desc() != _userSrcMem.get_desc())
				_convSrcMem = dnnl::memory(_convPd.src_desc(), _engine);

			_convWeightMem = _userWeightMem;
			if (_convPd.weights_desc() != _userWeightMem.get_desc())
			{
				_convWeightMem = dnnl::memory(_convPd.weights_desc(), _engine);
				dnnl::reorder(_userWeightMem, _convWeightMem).execute(_engine_stream, _userWeightMem, _convWeightMem);
				_engine_stream.wait();
			}

			_convDstMem = _userDstMem;
			if (_convPd.dst_desc() != _userDstMem.get_desc()) 
				_convDstMem = dnnl::memory(_convPd.dst_desc(), _engine);

			_convPrim = dnnl::convolution_forward(_convPd);

			_convArgs.insert({ DNNL_ARG_SRC, _convSrcMem });
			_convArgs.insert({ DNNL_ARG_WEIGHTS, _convWeightMem });
			_convArgs.insert({ DNNL_ARG_BIAS, _userBiasMem });
			_convArgs.insert({ DNNL_ARG_DST, _convDstMem });

			return true;
		}

		virtual bool SetSrc(const Tensor& src)
		{
			write_to_dnnl_memory((void*)src.Data<float>(), _userSrcMem);
			if (_convPd.src_desc() != _userSrcMem.get_desc())
			{
				dnnl::reorder(_userSrcMem, _convSrcMem).execute(_engine_stream, _userSrcMem, _convSrcMem);
				_engine_stream.wait();
			}
			return true;
		}

		virtual bool Run()
		{
			_convPrim.execute(_engine_stream, _convArgs);
			
			_engine_stream.wait();

			return true;
		}

		virtual bool GetDst(Tensor& dst)
		{
			if (_convPd.dst_desc() != _userDstMem.get_desc())
			{
				dnnl::reorder(_convDstMem, _userDstMem).execute(_engine_stream, _convDstMem, _userDstMem);
				_engine_stream.wait();
			}
			else
				_userDstMem = _convDstMem;

			read_from_dnnl_memory(dst.Data<float>(), _userDstMem);
			return true;
		}
	};


	//----------------------------------------------------------------------------------------------------

	bool Convolution32fTest(const Options& options, float eps, const ConvParam& p, TestConvolution32f &f1, TestConvolution32f &f2)
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

		for(double start = Cpl::Time(), current = start; current <= start + options.testTime; current = Cpl::Time())
		{
			CPL_PERF_BEGF(p.Description() + " " + f1.Name(), p.Flop());
  			f1.Run();
		}

		for (double start = Cpl::Time(), current = start; current <= start + options.testTime; current = Cpl::Time())
		{
			CPL_PERF_BEGF(p.Description() + " " + f2.Name(), p.Flop());
			f2.Run();
		}

		f1.GetDst(dst1);
		f2.GetDst(dst2);

		return Compare32f(dst1, dst2, eps, true, 64);
	}

	bool Convolution32fTest(const Options& options)
	{
		Size _0(0, 0), _1(1, 1), _2(2, 2), _3(3, 3), _4(4, 4), _5(5, 5), _6(6, 6), _7(7, 7);
		const SimdConvolutionActivationType aId = SimdConvolutionActivationIdentity, aRe = SimdConvolutionActivationRelu,
			aLr = SimdConvolutionActivationLeakyRelu, aRr = SimdConvolutionActivationRestrictRange, aPr = SimdConvolutionActivationPrelu,
			aEl = SimdConvolutionActivationElu, aHs = SimdConvolutionActivationHswish, aMi = SimdConvolutionActivationMish,
			aHi = SimdConvolutionActivationHardSigmoid, aSw = SimdConvolutionActivationSwish, aGe = SimdConvolutionActivationGelu;
		const SimdBool tF = SimdFalse, tT = SimdTrue;
		const SimdTensorDataType f32 = SimdTensorData32f, b16 = SimdTensorData16b;
		float eps = 0.05f, time = 0.01f;

		bool result = true;

		Cpl::PerformanceStorage::Global().Clear();

		result = result && Convolution32fTest(options, eps, ConvParam(1, 384, 13, 13, 1152, _1, _1, _1, _0, _0, 1, aRe, tF), TestConvolution32fSimd().Ref(), TestConvolution32fDnnl().Ref());
		result = result && Convolution32fTest(options, eps, ConvParam(1, 384, 13, 13, 1152, _1, _1, _1, _0, _0, 1, aRe, tT), TestConvolution32fSimd().Ref(), TestConvolution32fDnnl().Ref());

		CPL_LOG_SS(Info, std::endl << Cpl::PerformanceStorage::Global().Report());

		return result;
	}
}