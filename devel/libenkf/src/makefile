include path_config
SITE_CONFIG_FILE = "/h/a152128/EnKF/devel/EnKF/libenkf/src/site-config"
#m32 -> #define i386 1    m64 -> __x86_64 1
MFLAG = m32
CC = gcc
AR = ar
#################################################################
MAIN_INCLUDE     = -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBSCHED_HOME)/include -I$(LIBRMS_HOME)/include -I$(LIBECL_QUEUE_HOME)/include
COMPILE_INCLUDE  = -I$(LIBENKF_HOME)/src $(MAIN_INCLUDE)
INSTALL_INC_PATH =   $(LIBENKF_HOME)/include
INSTALL_LIB_PATH =   $(LIBENKF_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O3 -std=gnu99 -g -Wall 
ARFLAGS = -csr
#################################################################
OBJECTS           = field_interC.o thist_interC.o thist.o scalar.o scalar_config.o trans_func.o logmode.o multflt_interC.o multz_interC.o enkf_types.o enkf_obs.o obs_data.o obs_node.o well_obs.o field_obs.o meas_op.o meas_vector.o well_config.o well.o enkf_config_node.o field_config.o field.o ecl_static_kw_config.o ecl_static_kw.o equil_config.o equil.o enkf_state.o multflt.o enkf_util.o multflt_config.o enkf_main.o multz.o multz_config.o enkf_node.o pgbox_config.o pgbox.o enkf_fs.o gen_kw_config.o gen_kw.o gen_kw_interC.o enkf_fs.o basic_driver.o plain_driver.o analysis.o meas_matrix.o config.o plain_driver_static.o fs_index.o enkf_config.o plain_driver_parameter.o relperm.o relperm_config.o relperm_interC.o enkf_site_config.o 

INC_FILES_INSTALL = thist.h scalar.h scalar_config.h trans_func.h logmode.h enkf_obs.h obs_data.h obs_node.h well_obs.h meas_op.h meas_vector.h well_config.h well.h enkf_config_node.h field_obs.h field_config.h field.h enkf_macros.h ecl_static_kw_config.h ecl_static_kw.h equil_config.h equil.h enkf_state.h multflt.h enkf_util.h multflt_config.h enkf_main.h multz.h multz_config.h enkf_node.h pgbox_config.h pgbox.h enkf_fs.h gen_kw_config.h gen_kw.h enkf_types.h basic_driver.h plain_driver.h analysis.h meas_matrix.h config.h plain_driver_static.h fs_index.h enkf_config.h plain_driver_parameter.h relperm.h relperm_config.h enkf_site_config.h 

ifeq ($(MFLAG),m64)
   LSF_LIBS          = -lbat  -llsf -lnsl
else
   LSF_LIBS          = 
endif

LIB_LINK          = -lenkf_$(MFLAG) -lsched_$(MFLAG) -lrms_$(MFLAG) -lecl_queue_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lanalysis_$(MFLAG) -lm -lpthread -lz  -lifcore -lirc  -llapack -lblas $(LSF_LIBS) 
LIB_PATH          = -L$(LIBECL_QUEUE_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib -L$(LIBRMS_HOME)/lib -L$(LIBENKF_HOME)/lib -L$(LIBANALYSIS_HOME)/lib -L$(EXTERNAL_LIB_PATH) -L$(EXTERNAL_LIB_PATH)/ifort_$(MFLAG)
LIB               = libenkf_$(MFLAG).a
TARGET_INCLUDE    = -I$(INSTALL_INC_PATH) $(MAIN_INCLUDE)


INSTALL: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) -D'SITE_CONFIG_FILE=$(SITE_CONFIG_FILE)' $(TARGET_INCLUDE) main.c  -o enkf.x  $(LIB_PATH) $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE) havanna_example.c  -o hav.x  $(LIB_PATH) $(LIB_LINK) 
	#gcc $(CFLAGS) $(TARGET_INCLUDE) test.c  -o test.x  $(LIB_PATH) $(LIB_LINK) 
	#gcc $(CFLAGS) $(TARGET_INCLUDE) thist_test.c -o thist.x $(LIB_PATH) $(LIB_LINK)
	#gcc $(CFLAGS) $(TARGET_INCLUDE) field_test.c -o field.x $(LIB_PATH) $(LIB_LINK)
	#gcc $(CFLAGS) $(TARGET_INCLUDE) swcr_init.c  -o /d/proj/bg/enkf/bin/swcr_init.x  $(LIB_PATH) $(LIB_LINK)
	#gcc $(CFLAGS) $(TARGET_INCLUDE) relp_test.c  -o relp.x  $(LIB_PATH) $(LIB_LINK)


clean:
	rm -f *.o *~
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f libenkf_m*
	rm -f $(INSTALL_LIB_PATH)/$(LIB)


$(LIB): $(OBJECTS) 
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS) $(FORTRAN_OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@

thist_interC.o       : thist.o
thist.o              : thist.c thist.h
logmode.o            : logmode.c logmode.h enkf_types.h
multflt_interC.o     : multflt_interC.c multflt.o multflt_config.o scalar.o
multz_interC.o       : multz_interC.c multz.o multz_config.o scalar.o
relperm_interC.o     : relperm_interC.c relperm.o relperm_config.o scalar.o
scalar.o               : scalar.h scalar.c enkf_util.o scalar_config.o
enkf_types.o         : enkf_types.c enkf_types.h
enkf_obs.o           : enkf_obs.c enkf_obs.h enkf_main.o obs_node.o well_obs.o field_obs.o
obs_data.o           : obs_data.c obs_data.h
obs_node.o           : obs_node.c obs_node.h enkf_types.h
meas_vector.o          : meas_vector.c meas_vector.h enkf_util.o
well_obs.o           : well_obs.h well_obs.c meas_op.o
meas_op.o            : meas_op.c meas_op.h
well_config.o        : enkf_main.c enkf_main.h enkf_types.h enkf_macros.h enkf_config_node.o well_config.c well_config.h well_config_private.h
well.o               : well.c well.h well_config.o 
enkf_config_node.o   : enkf_types.h enkf_macros.h
field_config.o       : field_config.c field_config.h enkf_types.h
field.o              : field.c field.h field_config.o enkf_types.h enkf_macros.h
equil_config.o : enkf_util.h equil_config.c equil_config.h enkf_macros.h scalar_config.o
equil.o        : equil.c equil_config.o equil.h enkf_macros.h enkf_types.h
enkf_state.o   : enkf_util.h enkf_state.c enkf_state.h enkf_node.o enkf_macros.h field.o field_config.o multz.o multflt.o equil.o multz_config.o multflt_config.o equil_config.o well.o well_config.o enkf_fs.o analysis.o gen_kw.o enkf_config.o
multflt_config.o : multflt_config.c multflt_config.h enkf_macros.h
multflt.o      : enkf_util.h multflt.c multflt.h multflt_config.o enkf_util.o logmode.o
ecl_config.o   : enkf_util.h ecl_config.c ecl_config.h enkf_macros.h enkf_config_node.o
multz_config.o : enkf_util.h multz_config.c multz_config.h enkf_macros.h logmode.o
relperm_config.o : enkf_util.h relperm_config.c relperm_config.h enkf_macros.h logmode.o
multz.o        : enkf_util.h multz.c multz.h multz_config.o logmode.o
relperm.o        : enkf_util.h relperm.c relperm.h relperm_config.o logmode.o
enkf_node.o    : enkf_util.h enkf_node.h enkf_node.c enkf_macros.h
field_interC.o : field_interC.c field.o field_config.o
field_obs.o    : field_obs.c field_obs.h field.o field_config.o
scalar_config.o : scalar_config.c scalar_config.h enkf_macros.h trans_func.o
trans_func.o    : trans_func.h trans_func.h 
pgbox_config.o  : pgbox_config.c pgbox_config.h
pgbox.o         : pgbox.c pgbox.h pgbox_config.o
enkf_fs.o       : enkf_fs.c enkf_fs.h basic_driver.o plain_driver.o fs_index.o
basic_driver.o  : basic_driver.c basic_driver.h
plain_driver.o  : plain_driver.c plain_driver.h
meas_matrix.o   : meas_matrix.c meas_matrix.h 
gen_kw.o        : gen_kw.c gen_kw.h gen_kw_config.o
gen_kw_config.o : gen_kw_config.c gen_kw_config.h
config.o        : config.c config.h
plain_driver_static.o    : plain_driver_static.c plain_driver_static.h enkf_node.o
plain_driver_parameter.o : plain_driver_parameter.c plain_driver_parameter.h enkf_node.o
fs_index.i            : fs_index.h fs_index.c
enkf_config.o         : enkf_config.c enkf_config.h enkf_site_config.o enkf_site_config.o
enkf_site_config.o    : enkf_site_config.h enkf_site_config.c
enkf_main.o           : enkf_main.c enkf_main.h enkf_state.o enkf_config.o enkf_site_config.o
analysis.o            : analysis.c meas_matrix.o obs_data.o

