#include <filesystem>
#include <fstream>
#include <sys/wait.h>

#include "catch2/catch.hpp"
#include <ert/enkf/enkf_fs.hpp>
#include <ert/enkf/enkf_obs.hpp>
#include <ert/util/test_util.h>

#include "../tmpdir.hpp"
#include "ert/res_util/block_fs.hpp"

enkf_fs_type *enkf_fs_alloc_empty(const char *mount_point,
                                  unsigned ensemble_size, bool read_only);
void enkf_fs_umount(enkf_fs_type *fs);

void enkf_fs_init_path_fmt(enkf_fs_type *fs);


TEST_CASE("block_fs", "[enkf_fs]") {
    const int fsync_interval = 10;

    std::vector<char> random(1000);
    {
        std::ifstream s{"/dev/urandom"};
        s.read(random.data(), random.size());
    }

    GIVEN("A single read-write instance of block_fs") {
        WITH_TMPDIR;
        auto bfs = block_fs_mount("bfs", fsync_interval, false /* read-only */);

        WHEN("data is written to storage") {
            block_fs_fwrite_file(bfs, "FOO", random.data(), random.size());

            THEN("data exists") {
                REQUIRE(block_fs_has_file(bfs, "FOO"));
                REQUIRE(!block_fs_has_file(bfs, "BAR"));
            }

            AND_THEN("data can be read from the same instance") {
                auto buf = buffer_alloc(100);
                block_fs_fread_realloc_buffer(bfs, "FOO", buf);

                REQUIRE(random.size() == buffer_get_size(buf));
                REQUIRE(std::memcmp(random.data(), buffer_get_data(buf),
                                    random.size()) == 0);
                buffer_free(buf);
            }

            AND_WHEN("block_fs is closed and opened") {
                block_fs_close(bfs);
                bfs =
                    block_fs_mount("bfs", fsync_interval, true /* read-only */);

                THEN("writing new data results in exception") {
                    REQUIRE_THROWS_WITH(
                        (block_fs_fwrite_file(bfs, "name", random.data(),
                                              random.size())),
                        Catch::Contains(
                            "tried to write to read only filesystem"));
                }

                THEN("data exists") {
                    REQUIRE(block_fs_has_file(bfs, "FOO"));
                    REQUIRE(!block_fs_has_file(bfs, "BAR"));
                }

                AND_THEN("data can be read") {
                    auto buf = buffer_alloc(100);
                    block_fs_fread_realloc_buffer(bfs, "FOO", buf);

                    REQUIRE(random.size() == buffer_get_size(buf));
                    REQUIRE(std::memcmp(random.data(), buffer_get_data(buf),
                                        random.size()) == 0);
                    buffer_free(buf);
                }
            }
        }

        WHEN("data is written to storage twice") {
            const std::string expect1 = "foo";
            const std::string expect2 = "bar";
            block_fs_fwrite_file(bfs, "FOO", expect1.data(), expect1.size());

            /* Read it back */
            auto buf = buffer_alloc(100);
            block_fs_fread_realloc_buffer(bfs, "FOO", buf);
            REQUIRE(expect1.size() == buffer_get_size(buf));
            REQUIRE(std::memcmp(expect1.data(), buffer_get_data(buf),
                                expect1.size()) == 0);
            buffer_free(buf);

            /* Overwrite */
            block_fs_fwrite_file(bfs, "FOO", expect2.data(), expect2.size());

            AND_WHEN("block_fs is closed and opened") {
                block_fs_close(bfs);
                bfs =
                    block_fs_mount("bfs", fsync_interval, true /* read-only */);

                THEN("reading FOO fill return the overwritten data") {
                    auto buf = buffer_alloc(100);
                    block_fs_fread_realloc_buffer(bfs, "FOO", buf);
                    REQUIRE(expect2.size() == buffer_get_size(buf));
                    REQUIRE(std::memcmp(expect2.data(), buffer_get_data(buf),
                                        expect2.size()) == 0);
                    buffer_free(buf);
                }
            }
        }

        block_fs_close(bfs);
    }
}
