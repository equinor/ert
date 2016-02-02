/*
  Copyright 2015 Statoil ASA.

  This file is part of the Open Porous Media project (OPM).

  OPM is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  OPM is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with OPM.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <ert/util/test_util.h>
#include <ert/util/test_work_area.h>

#include <ert/ecl/Smspec.hpp>

void test_smspec() {
    std::string kw( "WWCT" );
    std::string wg( "OP1" );
    std::string gr( "WG1" );

    ERT::smspec_node well( ECL_SMSPEC_WELL_VAR, wg, kw );
    ERT::smspec_node group( ECL_SMSPEC_GROUP_VAR, gr, kw );

    test_assert_true(well.wgname() == wg);
    test_assert_true(group.wgname() == gr);
}

int main (int argc, char **argv) {
    test_smspec();
}
