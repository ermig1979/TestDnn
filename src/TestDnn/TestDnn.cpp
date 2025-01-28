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

#include "Types.h"
#include "Options.h"

namespace TestDnn
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
            bool result = group.test(options);
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
    TestDnn::Options options(argc, argv);

    if (options.help)
        return options.PrintHelp();

    Cpl::Log::Global().AddStdWriter(options.logLevel);
    if (!options.logFile.empty())
        Cpl::Log::Global().AddFileWriter(options.logLevel, options.logFile);
    Cpl::Log::Global().SetFlags(Cpl::Log::BashFlags);

    TestDnn::Groups groups;
    for (const TestDnn::Group& group : TestDnn::g_groups)
        if (TestDnn::Required(group, options))
            groups.push_back(group);
		
	//::setenv("OMP_NUM_THREADS", "1", 1);
	//::setenv("OMP_WAIT_POLICY", "PASSIVE", 1);

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

    return TestDnn::MakeTests(groups, options);
}