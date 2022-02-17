/*
   Copyright (C) 2011  Equinor ASA, Norway.

   The file 'active_list.h' is part of ERT - Ensemble based Reservoir Tool.

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

#ifndef ERT_ACTIVE_LIST_H
#define ERT_ACTIVE_LIST_H

#include <stdio.h>

#include <vector>

#include <ert/enkf/enkf_types.hpp>

/**
   This enum is used when we are setting up the dependencies between
   observations and variables. The modes all_active and inactive are
   sufficient information, for the values partly active we need
   additional information.

   The same type is used both for variables (PRESSURE/PORO/MULTZ/...)
   and for observations.
*/

typedef enum {
    ALL_ACTIVE =
    1, /* The variable/observation is fully active, i.e. all cells/all faults/all .. */
    PARTLY_ACTIVE =
    3 /* Partly active - must supply additonal type spesific information on what is active.*/
} active_mode_type;

class ActiveList {
public:
    const std::vector<int> &index_list() const;
    const int *active_list_get_active() const;
    active_mode_type getMode() const;
    int getActiveSize(int default_size) const;
    void add_index(int index);
    void summary_fprintf(const char *dataset_key, const char *key,
                         FILE *stream) const;
    bool operator==(const ActiveList &other) const;
    void print_self() const {
        printf("ActiveList::self = %p\n", this);
        printf("mode: %d \n", static_cast<int>(this->m_mode));
        printf("index_list: {");
        for (const auto& v : this->m_index_list)
            printf(" %d",v);
        printf("} \n");
    }

private:
    std::vector<int> m_index_list;
    active_mode_type m_mode = ALL_ACTIVE;
};

#endif
