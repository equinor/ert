#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include <cerrno>
#include <fcntl.h>
#include <poll.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

#include <ert/job_queue/spawn.hpp>

extern char **environ;

static bool is_executable(const char *path) {
    if (access(path, F_OK) == 0) {
        struct stat stat_buffer;
        if (stat(path, &stat_buffer) != 0)
            throw std::runtime_error("Unable to get file properties of " +
                                     std::string(path));
        if (S_ISREG(stat_buffer.st_mode)) {
            return (stat_buffer.st_mode & S_IXUSR);
        } else
            return false; // It is not a file.
    } else                // Entry does not exist - return false.
        return false;
}

static std::shared_ptr<posix_spawnattr_t>
create_spawnattr(posix_spawnattr_t *spawn_attr) {
    int status = posix_spawnattr_init(spawn_attr);
    if (status != 0) {
        throw std::runtime_error(
            "Unable to initialize posix_spawn attributes " +
            std::string(strerror(errno)));
    }
    return {spawn_attr, posix_spawnattr_destroy};
}

static void set_spawn_flags(std::shared_ptr<posix_spawnattr_t> attributes) {
    short flags;
    if (posix_spawnattr_getflags(attributes.get(), &flags) != 0) {
        throw std::runtime_error("Unable to get posix_spawn flags " +
                                 std::string(strerror(errno)));
    }
    flags |= POSIX_SPAWN_SETPGROUP;
    if (posix_spawnattr_setflags(attributes.get(), flags) != 0) {
        throw std::runtime_error("Unable to set posix_spawn flags " +
                                 std::string(strerror(errno)));
    }

    if (posix_spawnattr_setpgroup(attributes.get(), 0) != 0) {
        throw std::runtime_error("Unable to set posix_spawn pgroup " +
                                 std::string(strerror(errno)));
    }
}

static std::shared_ptr<posix_spawn_file_actions_t>
create_fileactions(posix_spawn_file_actions_t *file_actions) {
    if (posix_spawn_file_actions_init(file_actions) != 0) {
        throw std::runtime_error("Unable to set up file redirection due to " +
                                 std::string(strerror(errno)));
    }
    return {file_actions, posix_spawn_file_actions_destroy};
}

/*
  At least when Python versions newer than 2.7.9 are involved it seems
  to be necessary to protect the access to posix_spawn with a mutex.
*/
static pthread_mutex_t spawn_mutex = PTHREAD_MUTEX_INITIALIZER;

/**
  The spawn function will start a new process running
  @executable. The pid of the new process will be
  returned. Alternatively the spawn_blocking() function will
  block until the newlye created process has completed.
*/
pid_t spawn(std::vector<std::string> argv) {
    std::unique_ptr<char *[]> argvptr(new char *[argv.size() + 1]);
    for (int i = 0; i < argv.size(); i++) {
        argvptr[i] = argv[i].data();
    }
    argvptr[argv.size()] = nullptr;
    return spawn(argvptr.get());
}

pid_t spawn(char *const argv[]) {
    pid_t pid;
    posix_spawnattr_t _spawn_attr{};
    auto spawn_attr = create_spawnattr(&_spawn_attr);
    set_spawn_flags(spawn_attr);
    pthread_mutex_lock(&spawn_mutex);
    int status = 0;
    if (is_executable(argv[0])) {
        status = posix_spawn(&pid, argv[0], nullptr, spawn_attr.get(), argv,
                             environ);
    } else {
        // look for executable in path
        status = posix_spawnp(&pid, argv[0], nullptr, spawn_attr.get(), argv,
                              environ);
    }

    if (status != 0)
        throw std::runtime_error("Could not call " + std::string(argv[0]) +
                                 " due to " + std::string(strerror(errno)));
    pthread_mutex_unlock(&spawn_mutex);
    return pid;
}

static void add_close(std::shared_ptr<posix_spawn_file_actions_t> fa, int p) {
    if (posix_spawn_file_actions_addclose(fa.get(), p) != 0) {
        throw std::runtime_error(
            "Unable to add posix_spawn close file_action " +
            std::string(strerror(errno)));
    }
}

static void add_dup2(std::shared_ptr<posix_spawn_file_actions_t> fa, int p1,
                     int p2) {
    if (posix_spawn_file_actions_adddup2(fa.get(), p1, p2) != 0) {
        throw std::runtime_error("Unable to add dup2 file_action " +
                                 std::string(strerror(errno)));
    }
}

/**
  Will spawn a new process and wait for its completion. The exit
  status of the new process is returned, observe that exit status 127
  typically means 'File not found' - i.e. the @executable could not be
  found.
*/
spawn_result spawn_blocking(std::vector<std::string> argv) {
    std::unique_ptr<char *[]> argvptr(new char *[argv.size() + 1]);
    for (int i = 0; i < argv.size(); i++) {
        argvptr[i] = argv[i].data();
    }
    argvptr[argv.size()] = nullptr;
    return spawn_blocking(argvptr.get());
}

spawn_result spawn_blocking(char *const argv[]) {
    posix_spawnattr_t _spawn_attr{};
    posix_spawn_file_actions_t _file_actions{};
    auto spawn_attr = create_spawnattr(&_spawn_attr);
    auto file_actions = create_fileactions(&_file_actions);
    set_spawn_flags(spawn_attr);
    pthread_mutex_lock(&spawn_mutex);
    int pid = 0;
    int status = 0;

    int cout_pipe[2];
    int cerr_pipe[2];

    if (pipe(cout_pipe) || pipe(cerr_pipe))
        throw std::runtime_error("Error while creating pipe");

    posix_spawn_file_actions_init(file_actions.get());
    add_close(file_actions, cout_pipe[0]);
    add_close(file_actions, cerr_pipe[0]);
    add_dup2(file_actions, cout_pipe[1], 1);
    add_dup2(file_actions, cerr_pipe[1], 2);
    add_close(file_actions, cout_pipe[1]);
    add_close(file_actions, cerr_pipe[1]);

    if (is_executable(argv[0])) {
        status = posix_spawn(&pid, argv[0], file_actions.get(),
                             spawn_attr.get(), argv, environ);
    } else {
        // look for executable in path
        status = posix_spawnp(&pid, argv[0], file_actions.get(),
                              spawn_attr.get(), argv, environ);
    }

    if (status != 0)
        throw std::runtime_error("Could not call " + std::string(argv[0]) +
                                 " due to " + std::string(strerror(errno)));
    close(cout_pipe[1]), close(cerr_pipe[1]);

    std::string buffer(1024, ' ');
    std::stringstream out;
    std::stringstream err;
    std::vector<pollfd> plist = {{cout_pipe[0], POLLIN},
                                 {cerr_pipe[0], POLLIN}};
    while (poll(&plist[0], plist.size(), /*timeout*/ -1) > 0) {
        if (plist[0].revents & POLLIN) {
            size_t bytes_read = read(cout_pipe[0], &buffer[0], buffer.length());
            std::cout << buffer.substr(0, bytes_read) << "\n";
            out << buffer.substr(0, bytes_read) << "\n";
        } else if (plist[1].revents & POLLIN) {
            size_t bytes_read = read(cerr_pipe[0], &buffer[0], buffer.length());
            err << buffer.substr(0, bytes_read) << "\n";
        } else
            break; // nothing left to read
    }

    pthread_mutex_unlock(&spawn_mutex);
    waitpid(pid, &status, 0);
    return {status, out.str(), err.str()};
}
