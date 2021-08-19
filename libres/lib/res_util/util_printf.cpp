/*
  Copyright (C) 2018  Equinor ASA, Norway.

  The file 'util_printf.c' is part of ERT - Ensemble based Reservoir Tool.

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

#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#include <ert/util/util.h>
#include <ert/res_util/util_printf.hpp>

void util_fprintf_string(const char *s, int width_,
                         string_alignement_type alignement, FILE *stream) {
    char fmt[32];
    size_t i;
    size_t width = width_;
    if (alignement == left_pad) {
        i = 0;
        if (width > strlen(s)) {
            for (i = 0; i < (width - strlen(s)); i++)
                fputc(' ', stream);
        }
        fprintf(stream, "%s", s);
    } else if (alignement == right_pad) {
        sprintf(fmt, "%%-%lus", width);
        fprintf(stream, fmt, s);
    } else {
        int total_pad = width - strlen(s);
        int front_pad = total_pad / 2;
        int back_pad = total_pad - front_pad;
        int i;
        util_fprintf_string(s, front_pad + strlen(s), left_pad, stream);
        for (i = 0; i < back_pad; i++)
            fputc(' ', stream);
    }
}

void util_fprintf_double(double value, int width, int decimals, char base_fmt,
                         FILE *stream) {
    char *fmt = util_alloc_sprintf("%c%d.%d%c", '%', width, decimals, base_fmt);
    fprintf(stream, fmt, value);
    free(fmt);
}

void util_fprintf_int(int value, int width, FILE *stream) {
    char fmt[32];
    sprintf(fmt, "%%%dd", width);
    fprintf(stream, fmt, value);
}
