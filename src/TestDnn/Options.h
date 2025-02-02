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

namespace td
{
    struct Options : public Cpl::ArgsParser
    {
        bool help;
        Cpl::Log::Level logLevel;
        String logFile;
        Strings include, exclude;
        float testTime, compareThreshold;
        int litterCache;

        Options(int argc, char* argv[])
            : Cpl::ArgsParser(argc, argv, true)
        {
            help = HasArg("-h", "-?");
            logLevel = (Cpl::Log::Level)Cpl::ToVal<int>(GetArg2("-ll", "--logLevel", "4", false));
            logFile = GetArg2("-lf", "--logFile", "", false);
            include = GetArgs("-i", Strings(), false);
            exclude = GetArgs("-e", Strings(), false);
            testTime = Cpl::ToVal<float>(GetArg2("-tt", "--testTime", "0.1", false));
            compareThreshold = Cpl::ToVal<float>(GetArg2("-ct", "--compareThreshold", "0.001", false));
            testTime = Cpl::ToVal<float>(GetArg2("-tt", "--testTime", "0.1", false));
            litterCache = Cpl::ToVal<int>(GetArg2("-lc", "--litterCache", "0", false));
        }

        int PrintHelp()
        {
            std::cout << "Test DNN Project." << std::endl << std::endl;
            std::cout << "Test application parameters:" << std::endl << std::endl;
            std::cout << " -i=test      - include test filter." << std::endl << std::endl;
            std::cout << " -e=test      - exclude test filter." << std::endl << std::endl;
            std::cout << " -ll=1        - a log level." << std::endl << std::endl;
            std::cout << " -lf=test.log - a log file name." << std::endl << std::endl;
            std::cout << " -h or -?     - to print this help message." << std::endl << std::endl;
            std::cout << " -tt=0.1      - a test time in seconds." << std::endl << std::endl;
            std::cout << " -ct=0.001    - a frameworks output compare threshold." << std::endl << std::endl;
            std::cout << " -lc=0        - Fill a big array to litter CPU cache between test runs." << std::endl << std::endl;
            return 0;
        }
    };
}