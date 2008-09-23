include path_config
#m32 -> #define i386 1    m64 -> __x86_64 1
MFLAG = m64
CC    = gcc
AR    = ar
#################################################################
MAIN_INCLUDE     = -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBSCHED_HOME)/include -I$(LIBRMS_HOME)/include -I$(LIBJOB_QUEUE_HOME)/include -I$(EXTERNAL_INC_PATH) -I$(LIBPLOT_HOME)/include
COMPILE_INCLUDE  = -I$(LIBENKF_HOME)/src $(MAIN_INCLUDE)
INSTALL_INC_PATH =   $(LIBENKF_HOME)/include
INSTALL_LIB_PATH =   $(LIBENKF_HOME)/lib
#################################################################
__SDP_ROOT  	   = $(SDP_ROOT)/$(shell res_target.py)
SVN_VERSION 	   = $(shell svnversion ./)
COMPILE_TIME_STAMP = $(shell date)
CFLAGS      	   = -$(MFLAG) -O2 -std=gnu99 -g -Wall 
ARFLAGS     	   = -csr
LDFLAGS     	   = -rdynamic
#################################################################
OBJECTS           = thist.o scalar.o scalar_config.o trans_func.o logmode.o enkf_types.o enkf_obs.o obs_data.o obs_node.o well_obs.o field_obs.o meas_op.o meas_vector.o well_config.o well.o enkf_config_node.o field_config.o field.o ecl_static_kw.o equil_config.o equil.o enkf_state.o multflt.o enkf_util.o multflt_config.o enkf_main.o multz.o multz_config.o enkf_node.o enkf_fs.o gen_kw_config.o gen_kw.o enkf_fs.o basic_driver.o plain_driver_dynamic.o analysis.o meas_matrix.o fs_index.o enkf_config.o plain_driver_parameter.o relperm.o relperm_config.o enkf_site_config.o havana_fault.o havana_fault_config.o summary_obs.o summary.o summary_config.o gen_data_config.o gen_data.o gen_common.o gen_obs.o enkf_sched.o enkf_ui_main.o enkf_ui_run.o enkf_ui_export.o enkf_ui_util.o plain_driver_static.o enkf_ui_plot.o enkf_ui_init.o gen_param_config.o gen_param.o enkf_serialize.o plain_driver_common.o

INC_FILES_INSTALL = thist.h scalar.h scalar_config.h trans_func.h logmode.h enkf_obs.h obs_data.h obs_node.h well_obs.h meas_op.h meas_vector.h well_config.h well.h enkf_config_node.h field_obs.h field_config.h field.h enkf_macros.h ecl_static_kw.h equil_config.h equil.h enkf_state.h multflt.h enkf_util.h multflt_config.h enkf_main.h multz.h multz_config.h enkf_node.h enkf_fs.h gen_kw_config.h gen_kw.h enkf_types.h basic_driver.h plain_driver_dynamic.h analysis.h meas_matrix.h fs_index.h enkf_config.h plain_driver_parameter.h relperm.h relperm_config.h enkf_site_config.h havana_fault.h havana_fault_config.h summary_obs.h summary_config.h summary_config.o gen_data_config.h gen_data.h gen_common.h gen_obs.h enkf_sched.h enkf_ui_main.h enkf_ui_run.h enkf_ui_export.h enkf_ui_util.h plain_driver_static.h enkf_ui_plot.h enkf_ui_init.h gen_param_config.h gen_param.h fs_types.h enkf_serialize.h plain_driver_common.h

ifeq ($(USE_LSF_LIBRARY),TRUE)
   LSF_LIBS   = -lbat  -llsf -lnsl
else
   LSF_LIBS   = 
endif

LIB_LINK          = -lenkf_$(MFLAG) -lsched_$(MFLAG) -lrms_$(MFLAG) -ljob_queue_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lanalysis_$(MFLAG) -lplot_$(MFLAG) -lplplotd -lm -lpthread -lz  -lifcore -lirc  -llapack -lblas $(LSF_LIBS) 
LIB_PATH          = -L$(LIBJOB_QUEUE_HOME)/lib -L$(LIBPLOT_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib -L$(LIBRMS_HOME)/lib -L./ -L$(LIBANALYSIS_HOME)/lib -L$(EXTERNAL_LIB_PATH) -L$(EXTERNAL_LIB_PATH)/ifort_$(MFLAG)
LIB_ROOT          = enkf
LIB               = lib$(LIB_ROOT)_$(MFLAG).a
TARGET_INCLUDE    = -I$(INSTALL_INC_PATH) $(MAIN_INCLUDE)



LOCAL: LIB BIN
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

SDP_INSTALL: LIB BIN
	install $(LIB) $(__SDP_ROOT)/lib/lib$(LIB_ROOT).a
	install $(BIN_FILES_SDP_INSTALL) $(__SDP_ROOT)/bin
	install $(INC_FILES_INSTALL) $(__SDP_ROOT)/include

LIB:	$(LIB)

clean:
	rm -f *.o *~ 
	rm -f $(LIB)
	rm -f $(INSTALL_LIB_PATH)/$(LIB)
	rm -f $(INSTALL_INC_PATH)/*.h


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)


%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@


#################################################################

# Library objects:
havana_fault.o       : havana_fault.c havana_fault.h enkf_util.o enkf_types.h scalar.o 
thist.o              : thist.c thist.h
logmode.o            : logmode.c logmode.h enkf_types.h
scalar.o               : scalar.h scalar.c enkf_util.o scalar_config.o
enkf_types.o         : enkf_types.c enkf_types.h
enkf_obs.o           : enkf_obs.c enkf_obs.h obs_node.o well_obs.o field_obs.o summary_obs.o
obs_data.o           : obs_data.c obs_data.h
obs_node.o           : obs_node.c obs_node.h enkf_types.h enkf_util.o well_obs.o summary_obs.o gen_obs.o field_obs.o
meas_vector.o          : meas_vector.c meas_vector.h enkf_util.o
well_obs.o           : well_obs.h well_obs.c meas_op.o enkf_macros.h
meas_op.o            : meas_op.c meas_op.h
well_config.o        : enkf_types.h enkf_macros.h enkf_config_node.o well_config.c well_config.h well_config_private.h
well.o               : well.c well.h well_config.o 
enkf_config_node.o   : enkf_types.h enkf_macros.h
field_config.o       : field_config.c field_config.h enkf_types.h
field.o              : field.c field.h field_config.o enkf_types.h enkf_macros.h
equil_config.o   : enkf_util.h equil_config.c equil_config.h enkf_macros.h scalar_config.o
equil.o          : equil.c equil_config.o equil.h enkf_macros.h enkf_types.h
enkf_state.o     : enkf_util.h enkf_state.c enkf_state.h enkf_node.o enkf_macros.h field.o field_config.o multz.o multflt.o equil.o multz_config.o multflt_config.o equil_config.o well.o well_config.o enkf_fs.o analysis.o gen_kw.o enkf_config.o
multflt_config.o : multflt_config.c multflt_config.h enkf_macros.h
multflt.o        : enkf_util.h multflt.c multflt.h multflt_config.o enkf_util.o logmode.o scalar.o
ecl_config.o     : enkf_util.h ecl_config.c ecl_config.h enkf_macros.h enkf_config_node.o
multz_config.o   : enkf_util.h multz_config.c multz_config.h enkf_macros.h logmode.o
relperm_config.o : enkf_util.h relperm_config.c relperm_config.h enkf_macros.h logmode.o
multz.o          : enkf_util.h multz.c multz.h multz_config.o logmode.o scalar.o enkf_util.o
relperm.o        : enkf_util.h relperm.c relperm.h relperm_config.o logmode.o scalar.o enkf_macros.h
enkf_node.o      : enkf_util.h enkf_node.h enkf_node.c enkf_macros.h gen_param_config.o gen_param.o enkf_serialize.o havana_fault.o field.o gen_data.o well.o summary.o gen_kw.o multz.o multflt.o equil.o relperm.o
field_obs.o      : field_obs.c field_obs.h field.o field_config.o
scalar_config.o : scalar_config.c scalar_config.h enkf_macros.h trans_func.o
trans_func.o    : trans_func.h trans_func.h 
enkf_fs.o       : enkf_fs.c enkf_fs.h basic_driver.o plain_driver_dynamic.o fs_index.o plain_driver_parameter.o plain_driver_static.o 
basic_driver.o  : basic_driver.c basic_driver.h
plain_driver_dynamic.o  : plain_driver_dynamic.c plain_driver_dynamic.h fs_types.h plain_driver_common.h
meas_matrix.o   : meas_matrix.c meas_matrix.h 
gen_kw.o        : gen_kw.c gen_kw.h gen_kw_config.o enkf_macros.h
gen_kw_config.o : gen_kw_config.c gen_kw_config.h
plain_driver_parameter.o : plain_driver_parameter.c plain_driver_parameter.h enkf_node.o fs_types.h plain_driver_common.o 
plain_driver_static.o 	 : plain_driver_static.c plain_driver_static.h enkf_node.o fs_types.h plain_driver_common.o
plain_driver_common.o    : plain_driver_common.c plain_driver_common.h
fs_index.i            	 : fs_index.h fs_index.c
enkf_config.o         	 : enkf_config.c enkf_config.h enkf_site_config.o enkf_site_config.o
enkf_site_config.o    	 : enkf_site_config.h enkf_site_config.c
enkf_main.o           	 : enkf_main.c enkf_main.h enkf_state.o enkf_config.o enkf_site_config.o enkf_obs.o 
analysis.o            	 : analysis.c meas_matrix.o obs_data.o
summary_obs.o         	 : summary_obs.h summary_obs.c enkf_util.o
summary.o             	 : summary.c summary.h
summary_config.o      	 : summary_config.c summary_config.h
gen_common.o          	 : gen_common.c gen_common.h
gen_data_config.o     	 : gen_data_config.c gen_data_config.h
gen_data.o            	 : gen_data.c gen_data.h gen_data_config.o enkf_macros.h
gen_obs.o             	 : gen_obs.c gen_obs.h gen_data_config.o enkf_macros.h
main.o                	 : enkf_main.o enkf_types.o enkf_config.o enkf_fs.o fs_index.o enkf_sched.o
enkf_sched.o          	 : enkf_sched.c enkf_sched.h
enkf_ui_main.o        	 : enkf_ui_main.c enkf_ui_main.h enkf_ui_run.o enkf_ui_export.o enkf_ui_util.o enkf_ui_plot.o enkf_ui_init.o
enkf_ui_run.o         	 : enkf_ui_run.c enkf_ui_run.h enkf_ui_util.o
enkf_ui_export.o      	 : enkf_ui_export.c enkf_ui_export.h
enkf_ui_util.o        	 : enkf_ui_util.c enkf_ui_util.h
enkf_ui_plot.o	         : enkf_ui_plot.c enkf_ui_plot.h enkf_ui_util.o
enkf_ui_init.o           : enkf_ui_init.c enkf_ui_init.h
gen_param_config.o       : gen_param_config.c gen_param_config.h
gen_param.o              : gen_param.c gen_param.h gen_param_config.o enkf_macros.h enkf_util.o
enkf_serialize.o           : enkf_serialize.h enkf_serialize.c

#################################################################
# Binaries

BIN_FILES             = enkf gen_test.x field_convert.x mod_err.x
BIN:	                $(BIN_FILES)
BIN_FILES_SDP_INSTALL = enkf

main.o		: main.c


main.o	: main.c
	$(CC) -c $(CFLAGS) -D'SITE_CONFIG_FILE="$(SITE_CONFIG_FILE)"' -D'SVN_VERSION="$(SVN_VERSION)"' -D'COMPILE_TIME_STAMP="$(COMPILE_TIME_STAMP)"' $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

enkf:	main.o $(LIB)
	gcc -$(MFLAG) $(LDFLAGS) main.o -o enkf $(LIB_PATH)  $(LIB_LINK) 	

test.x:	test.o $(LIB)
	gcc -$(MFLAG) $(LDFLAGS) test.o -o test.x $(LIB_PATH)  $(LIB_LINK) 	

gen_test.x:	gen_test.o $(LIB)
	gcc -$(MFLAG) $(LDFLAGS) gen_test.o -o gen_test.x  $(LIB_PATH)  $(LIB_LINK) 	

field_convert.x:	field_convert.o $(LIB)
	gcc -$(MFLAG) $(LDFLAGS) field_convert.o -o field_convert.x  $(LIB_PATH)  $(LIB_LINK) 	

sched_test.x:	sched_test.o $(LIB)
	gcc -$(MFLAG) $(LDFLAGS) sched_test.o -o sched_test.x  $(LIB_PATH)  $(LIB_LINK) 	


#################################################################
# Seriously ugly stuff, remove when Sigurd's paper is done.
mod_err.x: mod_err.o $(LIB)
	gcc -$(MFLAG) $(LDFLAGS) mod_err.o -o mod_err.x  $(LIB_PATH) -L$(LIBSAMPLE_HOME)/lib  -lsample_$(MFLAG) -lfftw3_$(MFLAG) -limf $(LIB_LINK)
