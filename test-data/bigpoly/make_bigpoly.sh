#!/bin/bash
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cp -a ../poly_example/ .
polystub=poly_stub.ert
touch $polystub

# shellcheck disable=SC2130 # style
echo "JOBNAME bigpoly_%d" >> $polystub
echo "RUNPATH poly_out/realization-<IENS>/iter-<ITER>" >> $polystub
echo "OBS_CONFIG observations" >> $polystub
echo "MAX_SUBMIT 1" >> $polystub
echo "GEN_KW COEFFS coeff_priors" >> $polystub
echo "GEN_DATA POLY_RES RESULT_FILE:poly.out" >> $polystub
echo "INSTALL_JOB poly_eval POLY_EVAL" >> $polystub

echo "EXECUTABLE /usr/bin/env" > ENV
echo "INSTALL_JOB env ENV" >> $polystub


###########################################
echo "Making small poly"

polysmall=smallpoly.ert
cp $polystub $polysmall
echo "NUM_REALIZATIONS 2" >> $polysmall
echo "SIMULATION_JOB poly_eval" >> $polysmall

polysmall_local=smallpoly_local.ert

cp $polysmall $polysmall_local
echo "QUEUE_SYSTEM LOCAL" >> $polysmall_local
echo "QUEUE_OPTION LOCAL MAX_RUNNING 10" >> $polysmall_local


###########################################
echo "Making small poly with FIELD update"

polyfile_field=smallpoly_field.ert
cp $polystub $polyfile_field

# shellcheck disable=SC2129 # style
echo "NUM_REALIZATIONS 2" >> $polyfile_field
echo "INSTALL_JOB symlink_grdecl SYMLINK_GRDECL" >> $polyfile_field
echo "SIMULATION_JOB poly_eval" >> $polyfile_field
echo "SIMULATION_JOB symlink_grdecl" >> $polyfile_field
echo "GRID ERTBOX.EGRID" >> $polyfile_field
echo "FIELD F_PARAM PARAMETER fieldparam.grdecl INIT_FILES:fieldparam.grdecl INIT_TRANSFORM:LOG OUTPUT_TRANSFORM:EXP MIN:-5.5 MAX:5.5 FORWARD_INIT:True" >> $polyfile_field

cat > make_egrid.py << EOF
import xtgeo
grid = xtgeo.create_box_grid(dimension=(200, 200, 200))
grid.to_file("ERTBOX.EGRID", "egrid")
EOF

python make_egrid.py

cat > make_random_grdecl.py << EOF
#!/bin/env python
import numpy as np
values = np.random.uniform(size=200*200*200)
with open("fieldparam.grdecl", "w", encoding="utf-8") as filehandle:
    filehandle.write("F_PARAM\n")
    filehandle.write(" ".join([str(val) for val in values]) + " \n/\n")
EOF
python make_random_grdecl.py
# This produces a file fieldparam.grdecl which we can use as a
# static file in all realizations by symlinking

echo "#!/bin/bash" > symlink_grdecl.sh
echo "ln -sf ../../../fieldparam.grdecl" >> symlink_grdecl.sh
chmod a+x symlink_grdecl.sh
echo "EXECUTABLE symlink_grdecl.sh" > SYMLINK_GRDECL


polyfield_local=smallpoly_field_local.ert
cp smallpoly_field.ert $polyfield_local

echo "QUEUE_SYSTEM LOCAL" >> $polyfield_local
echo "QUEUE_OPTION LOCAL MAX_RUNNING 10" >> $polyfield_local


###########################################
echo "Generate random summary file"

python3 -m venv venv
source venv/bin/activate
pip install hypothesis resfo pydantic
python generate_eclsum.py BIGPOLY


#####################################################################################
echo "Make bigpoly (many realizations, many forward_models and with summary-file)"

polybig=bigpoly.ert

cp poly_stub.ert $polybig

num_real=100

# shellcheck disable=SC2129 # style
echo "NUM_REALIZATIONS $num_real" >> $polybig
echo "ECLBASE BIGPOLY" >> $polybig
echo "SUMMARY *" >> $polybig
echo "FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/BIGPOLY.SMSPEC,<TO>=BIGPOLY.SMSPEC)" >> $polybig
echo "FORWARD_MODEL COPY_FILE(<FROM>=<CONFIG_PATH>/BIGPOLY.UNSMRY,<TO>=BIGPOLY.UNSMRY)" >> $polybig

for _ in $(seq 0 $num_real); do
  echo "SIMULATION_JOB env" >> $polybig
done

echo "SIMULATION_JOB poly_eval" >> $polybig

for _ in $(seq 0 $num_real); do
  echo "SIMULATION_JOB env" >> $polybig
done
