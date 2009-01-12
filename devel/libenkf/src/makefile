include global_config
#m32 -> #define i386 1    m64 -> __x86_64 1
#################################################################
MAIN_INCLUDE     = -I$(LIBCONFIG_HOME)/include -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBSCHED_HOME)/include -I$(LIBRMS_HOME)/include -I$(LIBJOB_QUEUE_HOME)/include -I$(EXTERNAL_INC_PATH) -I$(LIBPLOT_HOME)/include
COMPILE_INCLUDE  = -I$(LIBENKF_HOME)/src $(MAIN_INCLUDE)
INSTALL_INC_PATH =   $(LIBENKF_HOME)/include
INSTALL_LIB_PATH =   $(LIBENKF_HOME)/lib
#################################################################
__SDP_ROOT  	   = $(SDP_ROOT)/$(shell res_target.py)
SVN_VERSION 	   = $(shell svnversion ./)
COMPILE_TIME_STAMP = $(shell date)
#################################################################
OBJECTS           = thist.o scalar.o scalar_config.o trans_func.o enkf_types.o enkf_obs.o obs_data.o field_obs.o meas_op.o meas_vector.o enkf_config_node.o field_config.o field.o ecl_static_kw.o equil_config.o equil.o enkf_state.o multflt.o enkf_util.o multflt_config.o enkf_main.o multz.o multz_config.o enkf_node.o enkf_fs.o gen_kw_config.o gen_kw.o enkf_fs.o basic_driver.o plain_driver_dynamic.o analysis.o meas_matrix.o plain_driver_parameter.o relperm.o relperm_config.o  havana_fault.o havana_fault_config.o summary_obs.o summary.o summary_config.o gen_data_config.o gen_data.o gen_common.o gen_obs.o enkf_sched.o enkf_ui_main.o enkf_ui_run.o enkf_ui_export.o enkf_ui_util.o plain_driver_static.o enkf_ui_plot.o enkf_ui_init.o enkf_serialize.o plain_driver_common.o ecl_config.o enkf_defaults.o ensemble_config.o model_config.o site_config.o pilot_point.o pilot_point_config.o active_node.o active_config.o gen_data_active.o gen_obs_active.o field_active.o active_list.o obs_vector.o field_trans.o plain_driver_index.o plain_driver.o

INC_FILES_INSTALL = thist.h scalar.h scalar_config.h trans_func.h enkf_obs.h obs_data.h meas_op.h meas_vector.h enkf_config_node.h field_obs.h field_config.h field.h enkf_macros.h ecl_static_kw.h equil_config.h equil.h enkf_state.h multflt.h enkf_util.h multflt_config.h enkf_main.h multz.h multz_config.h enkf_node.h enkf_fs.h gen_kw_config.h gen_kw.h enkf_types.h basic_driver.h plain_driver_dynamic.h analysis.h meas_matrix.h plain_driver_parameter.h relperm.h relperm_config.h  havana_fault.h havana_fault_config.h summary_obs.h summary_config.h summary_config.o gen_data_config.h gen_data.h gen_common.h gen_obs.h enkf_sched.h enkf_ui_main.h enkf_ui_run.h enkf_ui_export.h enkf_ui_util.h plain_driver_static.h enkf_ui_plot.h enkf_ui_init.h fs_types.h enkf_serialize.h plain_driver_common.h ecl_config.h ensemble_config.h model_config.h site_config.h pilot_point.h pilot_point_config.h active_node.h active_config.h gen_data_active.h gen_obs_active.h field_active.h active_list.h obs_vector.h field_trans.h plain_driver_index.h plain_driver.h


ifeq ($(USE_LSF_LIBRARY),TRUE)
   LSF_LIBS   = -lbat  -llsf -lnsl
else
   LSF_LIBS   = 
endif

LIB_LINK          = -lenkf_$(MFLAG) -lsched_$(MFLAG) -lrms_$(MFLAG) -ljob_queue_$(MFLAG) -lecl_$(MFLAG) -lconfig_$(MFLAG) -lutil_$(MFLAG) -lanalysis_$(MFLAG) -lplot_$(MFLAG) -lplplotd -lm -lpthread -lz  -llapack -lblas $(LSF_LIBS) -lgfortran 
LIB_PATH          = -L$(LIBJOB_QUEUE_HOME)/lib -L$(LIBPLOT_HOME)/lib -L$(LIBCONFIG_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib -L$(LIBRMS_HOME)/lib -L./ -L$(LIBANALYSIS_HOME)/lib -L$(EXTERNAL_LIB_PATH)
LIB_ROOT          = enkf
LIB               = lib$(LIB_ROOT)_$(MFLAG).a
TARGET_INCLUDE    = -I$(INSTALL_INC_PATH) $(MAIN_INCLUDE)



LOCAL: 	LIB BIN 
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

SDP_INSTALL: LIB BIN
	install $(LIB) $(__SDP_ROOT)/lib/lib$(LIB_ROOT).a
	install $(BIN_FILES_SDP_INSTALL) $(__SDP_ROOT)/bin
	install $(INC_FILES_INSTALL) $(__SDP_ROOT)/include


APPS:	LOCAL
	$(MAKE) -C $(APPLICATION_HOME)

LIB:	$(LIB)

clean_enkf: 
	rm -f $(BIN_FILES)


clean:  clean_enkf
	rm -f *.o *~ 
	rm -f $(LIB)
	rm -f $(INSTALL_LIB_PATH)/$(LIB)
	rm -f $(INSTALL_INC_PATH)/*.h

rebuild: clean LOCAL

new: 
	../../Scripts/cdep.py main.c

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)


%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@


#################################################################
include dependencies
#################################################################
# Binaries

BIN_FILES             = enkf gen_test.x test.x
BIN:	                $(BIN_FILES)
BIN_FILES_SDP_INSTALL = enkf

main.o		: main.c


main.o	: main.c
	$(CC) -c $(CFLAGS) -D'SITE_CONFIG_FILE="$(SITE_CONFIG_FILE)"' -D'SVN_VERSION="$(SVN_VERSION)"' -D'COMPILE_TIME_STAMP="$(COMPILE_TIME_STAMP)"' $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

enkf:	main.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) main.o -o enkf $(LIB_PATH)  $(LIB_LINK) 	

test.x:	test.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) test.o -o test.x $(LIB_PATH)  $(LIB_LINK) 	

gen_test.x:	gen_test.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) gen_test.o -o gen_test.x  $(LIB_PATH)  $(LIB_LINK) 	

field_convert.x:	field_convert.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) field_convert.o -o field_convert.x  $(LIB_PATH)  $(LIB_LINK) 	

sched_test.x:	sched_test.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) sched_test.o -o sched_test.x  $(LIB_PATH)  $(LIB_LINK) 	

