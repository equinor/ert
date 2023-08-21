#include <cstdlib>
#include <cstring>
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

extern char **environ;

static bool is_executable(const char *path) {
    if (access(path, F_OK) == 0) {
        struct stat stat_buffer {};
        stat(path, &stat_buffer);
        if (S_ISREG(stat_buffer.st_mode))
            return (stat_buffer.st_mode & S_IXUSR);
        else
            return false; // It is not a file.
    } else                // Entry does not exist - return false.
        return false;
}

static void spawn_init_attributes(posix_spawnattr_t *attributes) {
    posix_spawnattr_init(attributes);
    short flags;

    posix_spawnattr_getflags(attributes, &flags);
    flags |= POSIX_SPAWN_SETPGROUP;
    posix_spawnattr_setflags(attributes, flags);

    posix_spawnattr_setpgroup(attributes, 0);
}
static int spawn_init_redirection(posix_spawn_file_actions_t *file_actions,
                                  const char *stdout_file,
                                  const char *stderr_file) {
    int status = posix_spawn_file_actions_init(file_actions);

    if (status != 0) {
        throw std::runtime_error("Unable to set up file redirection due to " +
                                 std::string(strerror(errno)));
    }

    /* STDIN is unconditionally closed in the child process. */
    status = posix_spawn_file_actions_addclose(file_actions, STDIN_FILENO);

    if (status != 0) {
        throw std::runtime_error("Unable to set up file redirection due to " +
                                 std::string(strerror(errno)));
    }

    /* The _addopen() call will first close the fd and then reopen it;
     if no file is specified for stdout/stderr redirect the child will
     send stdout & stderr to whereever the parent process was already
     sending it.
    */
    if (stdout_file)
        status += posix_spawn_file_actions_addopen(
            file_actions, STDOUT_FILENO, stdout_file,
            O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IWUSR);

    if (stderr_file)
        status += posix_spawn_file_actions_addopen(
            file_actions, STDERR_FILENO, stderr_file,
            O_WRONLY | O_TRUNC | O_CREAT, S_IRUSR | S_IWUSR);

    return status;
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
    char **argv__ = (char **)malloc((argc + 2) * sizeof *argv__);

    {
        argv__[0] = (char *)executable;
        for (int iarg = 0; iarg < argc; iarg++)
            argv__[iarg + 1] = (char *)argv[iarg];
        argv__[argc + 1] = nullptr;
    }

    {
        posix_spawnattr_t spawn_attr;
        posix_spawn_file_actions_t file_actions;
        spawn_init_redirection(&file_actions, stdout_file, stderr_file);
        spawn_init_attributes(&spawn_attr);
        pthread_mutex_lock(&spawn_mutex);
        {
            int status = 0;
            if (is_executable(executable)) {
                status = posix_spawn(&pid, executable, &file_actions,
                                     &spawn_attr, argv__, environ);
            } else {
                // look for exectuable in path
                status = posix_spawnp(&pid, executable, &file_actions,
                                      &spawn_attr, argv__, environ);
            }

            if (status != 0)
                throw std::runtime_error("Could not call " +
                                         std::string(executable) + " due to " +
                                         std::string(strerror(errno)));
        }
        pthread_mutex_unlock(&spawn_mutex);
        posix_spawn_file_actions_destroy(&file_actions);
        posix_spawnattr_destroy(&spawn_attr);
    }

    free(argv__);
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
