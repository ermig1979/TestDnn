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
#include "Cpl/Table.h"

namespace td
{
    inline String ReportTable()
    {
        typedef Cpl::PerformanceStorage::PmPtr PmPtr;
        struct Cells
        {
            PmPtr dnnl;
            PmPtr simd;
            PmPtr amx;
        };
        typedef Cpl::PerformanceStorage::FunctionMap FuncMap;
        typedef std::map<String, Cells> TestMap;

        FuncMap merged = Cpl::PerformanceStorage::Global().Merged();
        TestMap tests;
        for (FuncMap::const_iterator function = merged.begin(); function != merged.end(); ++function)
        {
            const String& fullName = function->first;
            size_t beg = fullName.find("[");
            size_t end = fullName.find("]");
            String name = fullName.substr(beg, end - beg + 1);
            Cells& test = tests[name];
            if (fullName.find("Dnnl") != String::npos)
                test.dnnl = function->second;
            if (fullName.find("Simd") != String::npos)
                test.simd = function->second;
            if (fullName.find("Amx") != String::npos)
                test.amx = function->second;
        }

        Cpl::Table table(5, tests.size());
        table.SetHeader(0, "Test", true);
        table.SetHeader(1, "Dnnl", false);
        table.SetHeader(2, "Simd", true);
        table.SetHeader(3, "Amx", true);
        table.SetHeader(4, "ratio", true);
        size_t row = 0;
        for (TestMap::const_iterator test = tests.begin(); test != tests.end(); ++test, ++row)
        {
            table.SetCell(0, row, test->first);
            if (test->second.dnnl)
                table.SetCell(1, row, Cpl::ToStr(test->second.dnnl->GFlops(), 0));
            if (test->second.simd)
                table.SetCell(2, row, Cpl::ToStr(test->second.simd->GFlops(), 0));
            if (test->second.amx)
                table.SetCell(3, row, Cpl::ToStr(test->second.amx->GFlops(), 0));
            if (test->second.dnnl && test->second.amx)
                table.SetCell(4, row, Cpl::ToStr(test->second.amx->GFlops() / test->second.dnnl->GFlops(), 2));
            else if (test->second.dnnl && test->second.simd)
                table.SetCell(4, row, Cpl::ToStr(test->second.simd->GFlops() / test->second.dnnl->GFlops(), 2));
        }
        return table.GenerateText();
    }

}