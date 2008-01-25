include path_config
MFLAG = m32
CC = gcc
AR = ar
#################################################################
COMPILE_INCLUDE  = -I$(LIBENKF_HOME)/src -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBSCHED_HOME)/include -I$(LIBRMS_HOME)/include
INSTALL_INC_PATH =   $(LIBENKF_HOME)/include
INSTALL_LIB_PATH =   $(LIBENKF_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O3 -std=gnu99 -g -Wall 
ARFLAGS = -csr
#################################################################
FORTRAN_OBJECTS   = m_enkfX5.o mod_anafunc.o m_mean_preserving_rotation.o m_random.o m_multa.o m_randrot.o
OBJECTS           = field_interC.o thist_interC.o thist.o scalar.o scalar_config.o trans_func.o logmode.o multflt_interC.o multz_interC.o enkf_types.o enkf_obs.o obs_data.o obs_node.o well_obs.o field_obs.o meas_op.o meas_vector.o well_config.o well.o enkf_config_node.o field_config.o field.o ecl_static_kw_config.o ecl_static_kw.o equil_config.o equil.o enkf_state.o multflt.o enkf_util.o multflt_config.o enkf_config.o multz.o multz_config.o enkf_node.o pgbox_config.o pgbox.o enkf_fs.o gen_kw_config.o gen_kw.o gen_kw_interC.o enkf_fs.o basic_driver.o plain_driver.o analysis.o meas_matrix.o
INC_FILES_INSTALL = thist.h scalar.h scalar_config.h trans_func.h logmode.h enkf_obs.h obs_data.h obs_node.h well_obs.h meas_op.h meas_vector.h well_config.h well.h enkf_config_node.h field_obs.h field_config.h field.h enkf_macros.h ecl_static_kw_config.h ecl_static_kw.h equil_config.h equil.h enkf_state.h multflt.h enkf_util.h multflt_config.h enkf_config.h multz.h multz_config.h enkf_node.h pgbox_config.h pgbox.h enkf_fs.h gen_kw_config.h gen_kw.h enkf_types.h basic_driver.h plain_driver.h analysis.h meas_matrix.h

LIB_LINK          = -lenkf_$(MFLAG) -lsched_$(MFLAG) -lrms_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lm -lpthread -lz -lg2c -lifcore -llapack -lblas 
LIB_PATH          = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib -L$(LIBRMS_HOME)/lib -L$(LIBENKF_HOME)/lib -L/h/a152128/lib/linux -L/local/intel/compiler90/ia32/fc/lib
LIB               = libenkf_$(MFLAG).a
TARGET_INCLUDE    = -I$(LIBENKF_HOME)/include -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBSCHED_HOME)/include -I$(LIBRMS_HOME)/include


INSTALL: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) $(TARGET_INCLUDE) enkf_test.c  -o enkf.x  $(LIB_PATH) $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE) thist_test.c -o thist.x $(LIB_PATH) $(LIB_LINK)
	gcc $(CFLAGS) $(TARGET_INCLUDE) field_test.c -o field.x $(LIB_PATH) $(LIB_LINK)
	gcc $(CFLAGS) $(TARGET_INCLUDE) swcr_init.c  -o /d/proj/bg/enkf/bin/swcr_init.x  $(LIB_PATH) $(LIB_LINK)


clean:
	rm -f *.o *~
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f libenkf_m*
	rm -f $(INSTALL_LIB_PATH)/$(LIB)
	ln -s ../../SOURCE/*.o .


$(LIB): $(OBJECTS) $(FORTRAN_OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS) $(FORTRAN_OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@

thist_interC.o       : thist.o
thist.o              : thist.c thist.h
logmode.o            : logmode.c logmode.h enkf_types.h
multflt_interC.o     : multflt_interC.c multflt.o multflt_config.o scalar.o
multz_interC.o       : multz_interC.c multz.o multz_config.o scalar.o
scalar.o               : scalar.h scalar.c enkf_util.o scalar_config.o
enkf_types.o         : enkf_types.c enkf_types.h
enkf_obs.o           : enkf_obs.c enkf_obs.h enkf_config.o obs_node.o well_obs.o field_obs.o
obs_data.o           : obs_data.c obs_data.h
obs_node.o           : obs_node.c obs_node.h enkf_types.h
meas_vector.o          : meas_vector.c meas_vector.h enkf_util.o
well_obs.o           : well_obs.h well_obs.c meas_op.o
meas_op.o            : meas_op.c meas_op.h
well_config.o        : enkf_config.c enkf_config.h enkf_types.h enkf_macros.h enkf_config_node.o well_config.c well_config.h well_config_private.h
well.o               : well.c well.h well_config.o 
enkf_config_node.o   : enkf_config.c enkf_config.h enkf_types.h enkf_macros.h
field_config.o       : field_config.c field_config.h enkf_types.h
field.o              : field.c field.h field_config.o enkf_types.h enkf_macros.h
equil_config.o : enkf_util.h equil_config.c equil_config.h enkf_macros.h scalar_config.o
equil.o        : equil.c equil_config.o equil.h enkf_macros.h enkf_types.h
enkf_state.o   : enkf_util.h enkf_state.c enkf_state.h enkf_node.o enkf_macros.h field.o field_config.o multz.o multflt.o equil.o multz_config.o multflt_config.o equil_config.o well.o well_config.o enkf_fs.o analysis.o
multflt_config.o : multflt_config.c multflt_config.h enkf_macros.h
multflt.o      : enkf_util.h multflt.c multflt.h multflt_config.o enkf_util.o logmode.o
ecl_config.o   : enkf_util.h ecl_config.c ecl_config.h enkf_macros.h enkf_config_node.o
multz_config.o : enkf_util.h multz_config.c multz_config.h enkf_macros.h logmode.o
multz.o        : enkf_util.h multz.c multz.h multz_config.o logmode.o
enkf_node.o    : enkf_util.h enkf_node.h enkf_node.c enkf_macros.h
field_interC.o : field_interC.c field.o field_config.o
field_obs.o    : field_obs.c field_obs.h field.o field_config.o
scalar_config.o : scalar_config.c scalar_config.h enkf_macros.h trans_func.o
trans_func.o    : trans_func.h trans_func.h 
pgbox_config.o  : pgbox_config.c pgbox_config.h
pgbox.o         : pgbox.c pgbox.h pgbox_config.o
enkf_fs.o       : enkf_fs.c enkf_fs.h basic_driver.o plain_driver.o
basic_driver.o  : basic_driver.c basic_driver.h
plain_driver.o  : plain_driver.c plain_driver.h
meas_matrix.o   : meas_matrix.c meas_matrix.h 

