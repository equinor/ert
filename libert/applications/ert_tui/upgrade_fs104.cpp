/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'main.c' is part of ERT - Ensemble based Reservoir Tool.

   ERT is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   ERT is distributed in the hope that it will be useful, but WITHOUT ANY
   WARRANTY; without even the implied warranty of MERCHANTABILITY or
   FITNESS FOR A PARTICULAR PURPOSE.

   See the GNU General Public License at <http://www.gnu.org/licenses/gpl.html>
   for more details.
*/
#include <signal.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <util.h>
#include <hash.h>
#include <stringlist.h>
#include <block_fs.h>
#include <msg.h>

#include <ecl_sum.h>
#include <ecl_smspec.h>

#include <config.h>

#include <local_driver.h>
#include <lsf_driver.h>
#include <ext_joblist.h>

