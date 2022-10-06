#include <stdlib.h>
#include <string.h>

#include <ert/util/test_util.hpp>
#include <ert/util/vector.hpp>

#include <ert/res_util/res_env.hpp>

int main(int argc, char **argv) {
    setenv("PATH", "/usr/bin:/bin:/usr/local/bin", 1);
    auto path_list = res_env_alloc_PATH_list();
    if (path_list[0].compare("/usr/bin"))
        test_error_exit("Failed on first path element\n");

    if (path_list[1].compare("/bin"))
        test_error_exit("Failed on second path element\n");

    if (path_list[2].compare("/usr/local/bin"))
        test_error_exit("Failed on third  path element\n");

    exit(0);
}
