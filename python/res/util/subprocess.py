#  Copyright (C) 2019  Equinor ASA, Norway.
#
#  The file 'subprocess.py' is part of ERT - Ensemble based Reservoir Tool.
#
#  ERT is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  ERT is distributed in the hope that it will be useful, but WITHOUT ANY
#  WARRANTY; without even the implied warranty of MERCHANTABILITY or
#  FITNESS FOR A PARTICULAR PURPOSE.
#
#  See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
#  for more details.


import os


def await_process_tee(process, *out_files):
    """Wait for process to finish, "tee"-ing the subprocess' stdout into all the
    given file objects.

    """
    out_fds = [f.fileno() for f in out_files]
    process_fd = process.stdout.fileno()

    while process.poll() is None:
        while True:
            bytes_ = os.read(process_fd, 4096)
            if bytes_ == b"":  # check EOF
                break
            for fd in out_fds:
                os.write(fd, bytes_)
    process.stdout.close()

    return process.returncode
