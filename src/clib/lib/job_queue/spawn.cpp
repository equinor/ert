#include <cstdlib>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <cerrno>
#include <fcntl.h>
#include <pthread.h>
#include <spawn.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

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

static void
spawn_init_redirection(std::shared_ptr<posix_spawn_file_actions_t> file_actions,
                       const char *stdout_file, const char *stderr_file) {

    /* STDIN is unconditionally closed in the child process. */
    if (posix_spawn_file_actions_addclose(file_actions.get(), STDIN_FILENO) !=
        0) {
        throw std::runtime_error("Unable to set up file redirection due to " +
                                 std::string(strerror(errno)));
    }

    /* The _addopen() call will first close the fd and then reopen it;
     if no file is specified for stdout/stderr redirect the child will
     send stdout & stderr to whereever the parent process was already
     sending it.
    */
    if (stdout_file)
        if (posix_spawn_file_actions_addopen(
                file_actions.get(), STDOUT_FILENO, stdout_file,
                O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IWUSR) != 0) {
            throw std::runtime_error(
                "Unable to add posix_spawn stdout file_action " +
                std::string(strerror(errno)));
        }

    if (stderr_file)
        if (posix_spawn_file_actions_addopen(
                file_actions.get(), STDERR_FILENO, stderr_file,
                O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IWUSR) != 0) {
            throw std::runtime_error(
                "Unable to add posix_spawn stderr file_action " +
                std::string(strerror(errno)));
        }
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
pid_t spawn(const char *executable, int argc, const char **argv,
            const char *stdout_file, const char *stderr_file) {
    pid_t pid;
    std::unique_ptr<char *[]> args(new char *[argc + 2]);

    {
        args[0] = (char *)executable;
        for (int iarg = 0; iarg < argc; iarg++)
            args[iarg + 1] = (char *)argv[iarg];
        args[argc + 1] = nullptr;
    }

    {
        posix_spawnattr_t _spawn_attr{};
        posix_spawn_file_actions_t _file_actions{};
        auto spawn_attr = create_spawnattr(&_spawn_attr);
        auto file_actions = create_fileactions(&_file_actions);
        spawn_init_redirection(file_actions, stdout_file, stderr_file);
        set_spawn_flags(spawn_attr);
        pthread_mutex_lock(&spawn_mutex);
        {
            int status = 0;
            if (is_executable(executable)) {
                status = posix_spawn(&pid, executable, file_actions.get(),
                                     spawn_attr.get(), args.get(), environ);
            } else {
                // look for executable in path
                status = posix_spawnp(&pid, executable, file_actions.get(),
                                      spawn_attr.get(), args.get(), environ);
            }

            if (status != 0)
                throw std::runtime_error("Could not call " +
                                         std::string(executable) + " due to " +
                                         std::string(strerror(errno)));
        }
        pthread_mutex_unlock(&spawn_mutex);
    }
    return pid;
}

/**
  Will spawn a new process and wait for its completion. The exit
  status of the new process is returned, observe that exit status 127
  typically means 'File not found' - i.e. the @executable could not be
  found.
*/
int spawn_blocking(const char *executable, int argc, const char **argv,
                   const char *stdout_file, const char *stderr_file) {
    pid_t pid = spawn(executable, argc, argv, stdout_file, stderr_file);
    int status;
    waitpid(pid, &status, 0);
    return status;
}
