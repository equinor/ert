import os


def await_process_tee(process, *out_files):
    """Wait for process to finish, "tee"-ing the subprocess' stdout into all the
    given file objects.

    NB: We aren't checking if `os.write` succeeds. It succeeds if its return
    value matches `len(bytes_)`. In other cases we might want to do something
    smart, such as retry or raise an error. At the time of writing it is
    uncertain what we should do, and it is assumed that data loss is acceptable.

    """
    out_fds = [f.fileno() for f in out_files]
    process_fd = process.stdout.fileno()

    while True:
        while True:
            bytes_ = os.read(process_fd, 4096)
            if bytes_ == b"":  # check EOF
                break
            for fd in out_fds:
                os.write(fd, bytes_)

        # Check if process terminated
        if process.poll() is not None:
            break
    process.stdout.close()

    return process.returncode
