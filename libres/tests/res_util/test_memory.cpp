#include <sstream>
#include <fmt/format.h>
#include "catch2/catch.hpp"

#include <ert/logging.hpp>
#include <ert/res_util/memory.hpp>

static int callno = 0;

/*
 * "Mock" the internal function which reads memory-info from
 * the /proc-filesystem.
 *
 * Note that although the real memory-info is only available
 * on Linux, this "mocking" makes the test run also on MacOs
 */
namespace ert {
namespace utils {
std::shared_ptr<std::istream> get_file(const char *filename) {
    std::string s = R"(
Name:	test
Umask:	0022
State:	R (running)
Tgid:	72786
Ngid:	0
Pid:	72786
PPid:	1
TracerPid:	0
Uid:	0	0	0	0
Gid:	0	0	0	0
FDSize:	256
Groups:
NStgid:	72786
NSpid:	72786
NSpgid:	72786
NSsid:	1
VmPeak:	  160560 kB
VmSize:	  146680 kB
VmLck:	       0 kB
VmPin:	       0 kB
VmHWM:	    7892 kB
VmRSS:	    7892 kB
RssAnon:	    4680 kB
RssFile:	    3212 kB
RssShmem:	       0 kB
VmData:	  141976 kB
VmStk:	     132 kB
VmExe:	    1628 kB
VmLib:	       4 kB
VmPTE:	      68 kB
VmSwap:	       0 kB
nonvoluntary_ctxt_switches:	4)";

    /**
    * This method is called four times while the test runs.
	* The last two calls are upon exiting the scope,
	* hence simulate changes
	**/
    if (callno > 1) {
        std::string find1("160560 kB"), find2("146680 kB");
        std::string replace1("260560 kB"), replace2("46680 kB");

        // Pretend VmPeak increased
        s.replace(s.find(find1), find1.size(), replace1);
        // Pretend VmSize decreased
        s.replace(s.find(find2), find2.size(), replace2);
    }
    callno++;

    return std::make_shared<std::istringstream>(s);
}
} // namespace utils
} // namespace ert

class MockLogger : public ert::ILogger {
public:
    std::vector<std::string> calls;

protected:
    void log(ert::ILogger::Level level, fmt::string_view f,
             fmt::format_args args) override {
        std::string s = fmt::vformat(f, args);
        calls.push_back(s);
    }
};

TEST_CASE("simple memory logger test", "[res_util]") {
    auto logger = std::make_shared<MockLogger>();
    { ert::utils::scoped_memory_logger memlogger(logger, "test"); }

    /* Dbl-check that the correct number of "mocked" calls were made */
    REQUIRE(callno == 4);

    /*
	 * The logger should now hold the following two strings
	 * 1. "Enter test Mem: 146Mb  MaxMem: 160Mb"
	 * 2. "Exit  test Mem: 46Mb (-100) MaxMem: 260Mb (+100)"
	 */
    REQUIRE(logger->calls.size() == 2);
    REQUIRE(logger->calls[0].find("Mem: 146Mb  MaxMem: 160Mb") !=
            std::string::npos);
    REQUIRE(logger->calls[1].find("(-100)") != std::string::npos);
    REQUIRE(logger->calls[1].find("(+100)") != std::string::npos);
}
