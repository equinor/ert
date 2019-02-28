/*
  Copyright (C) 2018  Equinor ASA, Norway.

  The file 'res_portability.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <pthread.h>

#include <ert/util/util.hpp>

#include <ert/res_util/res_portability.hpp>
#include "ert/util/build_config.h"

void res_yield() {
#ifdef HAVE_YIELD_NP
  pthread_yield_np();
#else
  #ifdef HAVE_YIELD
  pthread_yield();
  #else
  util_usleep(1000);
  #endif
#endif
}

