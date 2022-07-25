#include <chrono>
#include <regex>
#include <thread>

#include "catch2/catch.hpp"

#include "../logger_mock.hpp"
#include <ert/res_util/metric.hpp>

TEST_CASE("simple time benchmark test", "[res_util]") {
    auto logger = std::make_shared<MockLogger>();
    {
        ert::utils::Benchmark benchmark(logger, "some_function");
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    /**
	 * The logger should now hold the following string
	 *  "some_function's execution time: 2.xxxx seconds"
	 */
    REQUIRE(logger->calls.size() == 1);
    REQUIRE(std::regex_search(logger->calls[0], std::regex("2\\.\\d{4}")));
    REQUIRE(logger->calls[0].find("some_function's") != std::string::npos);
}
