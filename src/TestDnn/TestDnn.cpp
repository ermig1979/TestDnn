/*
* Test DNN Project (http://github.com/ermig1979/td).
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

#include "Types.h"
#include "Options.h"

#if defined(__linux__)
#include <signal.h>
#include <setjmp.h>
#endif

namespace td
{
    typedef bool(*TestPtr)(const Options & options);

    struct Group
    {
        String name;
        TestPtr test;

        Group(const String& n, const TestPtr& t)
            : name(n)
            , test(t)
        {
        }
    };
    typedef std::vector<Group> Groups;
    Groups g_groups;

#define TEST_ADD(name) \
    bool name##Test(const Options & options); \
    bool name##AddToList(){ g_groups.push_back(Group(#name, name##Test)); return true; } \
    bool name##AtList = name##AddToList();

    TEST_ADD(Convolution32f);
    TEST_ADD(Convolution16b);

    //-------------------------------------------------------------------------------------------------

#if defined(__linux__)
    static __thread jmp_buf s_threadData;
#endif

#if defined(__linux__)
    static void PrintErrorMessage(int code)
    {
        String desc;
        switch (code)
        {
        case SIGILL: desc = "Illegal instruction"; break;
        case SIGABRT: desc = "Aborted"; break;
        case SIGSEGV: desc = "Segment violation"; break;
        case SIGCHLD: desc = "Child exited"; break;
        default:
            desc = "Unknown error(" + std::to_string(code) + ")";
        }
        CPL_LOG_SS(Error, "There is unhandled Linux signal: " << desc << " !");
        longjmp(s_threadData, 1);
    }
#endif

    static bool RunGroup(const Group& group, const Options& options)
    {
#if defined(__linux__)
        std::vector<int> types;
        std::vector<__sighandler_t> prevs;
        for (int i = 0; i <= SIGSYS; ++i)
        {
            if (i == SIGCHLD)
                continue;
            __sighandler_t prev = signal(i, (__sighandler_t)PrintErrorMessage);
            if (prev == SIG_IGN)
                signal(i, prev);
            else
            {
                types.push_back(i);
                prevs.push_back(prev);
            }
        }
        int rc = setjmp(s_threadData);
        bool result = false;
        if (rc == 0)
            result = group.test(options);
        for (size_t i = 0; i < prevs.size(); ++i)
            signal(types[i], prevs[i]);
        return result;
#else
        return group.test(options);
#endif
    }

    //-------------------------------------------------------------------------------------------------

    bool Required(const Group& group, const Options& options)
    {
        bool required = options.include.empty();
        for (size_t i = 0; i < options.include.size() && !required; ++i)
            if (group.name.find(options.include[i]) != std::string::npos)
                required = true;
        for (size_t i = 0; i < options.exclude.size() && required; ++i)
            if (group.name.find(options.exclude[i]) != std::string::npos)
                required = false;
        return required;
    }

    int MakeTests(const Groups& groups, const Options& options)
    {
        for (size_t t = 0; t < groups.size(); ++t)
        {
            const Group& group = groups[t];
            CPL_LOG_SS(Info, group.name << "Test is started :");
            bool result = RunGroup(group, options);
            if (result)
            {
                CPL_LOG_SS(Info, group.name << "Test is OK." << std::endl);
            }
            else
            {
                CPL_LOG_SS(Error, group.name << "Test has errors. TEST EXECUTION IS TERMINATED!" << std::endl);
                return 1;
            }
        }
        CPL_LOG_SS(Info, "ALL TESTS ARE FINISHED SUCCESSFULLY!" << std::endl);
        return 0;
    }
}

int main(int argc, char* argv[])
{
    td::Options options(argc, argv);

    if (options.help)
        return options.PrintHelp();

    Cpl::Log::Global().AddStdWriter(options.logLevel);
    if (!options.logFile.empty())
        Cpl::Log::Global().AddFileWriter(options.logLevel, options.logFile);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

	//::setenv("OMP_NUM_THREADS", "1", 1);
	//::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);
    //::setenv("DNNL_VERBOSE", "1", 1);
    //OMP_NUM_THREADS=1 OMP_WAIT_POLICY=PASSIVE DNNL_VERBOSE=0 

    td::Groups groups;
    for (const td::Group& group : td::g_groups)
        if (td::Required(group, options))
            groups.push_back(group);
		
    if (groups.empty())
    {
        std::stringstream ss;
        ss << "There are not any suitable tests for current filters! " << std::endl;
        ss << "  Include filters: " << std::endl;
        for (size_t i = 0; i < options.include.size(); ++i)
            ss << "'" << options.include[i] << "' ";
        ss << std::endl;
        ss << "  Exclude filters: " << std::endl;
        for (size_t i = 0; i < options.exclude.size(); ++i)
            ss << "'" << options.exclude[i] << "' ";
        ss << std::endl;
        CPL_LOG_SS(Error, ss.str());
        return 1;
    }

    return td::MakeTests(groups, options);
}