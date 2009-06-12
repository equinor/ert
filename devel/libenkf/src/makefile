include global_config
#m32 -> #define i386 1    m64 -> __x86_64 1
#################################################################
MAIN_INCLUDE     = -I$(LIBCONFIG_HOME)/include -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBSCHED_HOME)/include -I$(LIBRMS_HOME)/include -I$(LIBJOB_QUEUE_HOME)/include -I$(LIBPLOT_HOME)/include -I$(EXTERNAL_INC_PATH) 
COMPILE_INCLUDE  = -I$(LIBENKF_HOME)/src $(MAIN_INCLUDE) 
INSTALL_INC_PATH =   $(LIBENKF_HOME)/include
INSTALL_LIB_PATH =   $(LIBENKF_HOME)/lib
#################################################################
SDP_ROOT  	   = $(shell get_sdp_root.py)
SVN_VERSION 	   = $(shell svnversion ./)
COMPILE_TIME_STAMP = $(shell date)
#################################################################
OBJECTS           = scalar.o scalar_config.o trans_func.o enkf_types.o enkf_obs.o obs_data.o field_obs.o meas_vector.o enkf_config_node.o field_config.o field.o ecl_static_kw.o enkf_state.o enkf_util.o enkf_main.o enkf_node.o enkf_fs.o gen_kw_config.o gen_kw.o enkf_fs.o basic_driver.o old_analysis.o meas_matrix.o summary_obs.o summary.o summary_config.o gen_data_config.o gen_data.o gen_common.o gen_obs.o enkf_sched.o enkf_ui_main.o enkf_ui_run.o enkf_ui_export.o enkf_ui_util.o enkf_ui_plot.o enkf_ui_init.o enkf_serialize.o plain_driver_common.o ecl_config.o enkf_defaults.o ensemble_config.o model_config.o site_config.o pilot_point.o pilot_point_config.o gen_data_active.o gen_obs_active.o field_active.o active_list.o obs_vector.o field_trans.o plain_driver_index.o plain_driver.o enkf_ui_fs.o enkf_ui_table.o havana_fault.o havana_fault_config.o multflt_config.o multflt.o sqlite3.o sqlite3_driver.o config_parser.o enkf_analysis.o local_ministep.o local_reportstep.o local_config.o analysis_config.o

INC_FILES_INSTALL = scalar.h scalar_config.h trans_func.h enkf_obs.h obs_data.h meas_vector.h enkf_config_node.h field_obs.h field_config.h field.h enkf_macros.h ecl_static_kw.h enkf_state.h enkf_util.h enkf_main.h enkf_node.h enkf_fs.h gen_kw_config.h gen_kw.h enkf_types.h basic_driver.h old_analysis.h meas_matrix.h summary_obs.h summary_config.h summary_config.o gen_data_config.h gen_data.h gen_common.h gen_obs.h enkf_sched.h enkf_ui_main.h enkf_ui_run.h enkf_ui_export.h enkf_ui_util.h enkf_ui_plot.h enkf_ui_init.h fs_types.h enkf_serialize.h plain_driver_common.h ecl_config.h ensemble_config.h model_config.h site_config.h pilot_point.h pilot_point_config.h gen_data_active.h gen_obs_active.h field_active.h active_list.h obs_vector.h field_trans.h plain_driver_index.h plain_driver.h enkf_ui_fs.h enkf_ui_table.h havana_fault.h havana_fault_config.h multflt_config.h multflt.h sqlite3.h sqlite3_driver.h config_parser.h enkf_analysis.h local_ministep.h local_reportstep.h local_config.h analysis_config.h


ifeq ($(USE_LSF_LIBRARY),TRUE)
   LSF_LIBS   = -lbat  -llsf -lnsl
else
   LSF_LIBS   = 
endif

LIB_LINK          = -lenkf -lsched -lrms -ljob_queue -lecl -lconfig -lutil -lanalysis -lplot -lplplotd -lm -lpthread -lz  -llapack -lblas $(LSF_LIBS) -lgfortran 
LIB_PATH          = -L$(LIBJOB_QUEUE_HOME)/lib -L$(LIBPLOT_HOME)/lib -L$(LIBCONFIG_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib -L$(LIBRMS_HOME)/lib -L./ -L$(LIBANALYSIS_HOME)/lib -L$(EXTERNAL_LIB_PATH) 
LIB               = libenkf.a
TARGET_INCLUDE    = -I$(INSTALL_INC_PATH) $(MAIN_INCLUDE)



LOCAL: 	LIB BIN 
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

SDP_INSTALL: LIB BIN
	install $(LIB) $(SDP_ROOT)/lib/lib$(LIB_ROOT).a
	install $(BIN_FILES_SDP_INSTALL) $(SDP_ROOT)/bin
	install $(INC_FILES_INSTALL) $(SDP_ROOT)/include


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

rebuild: clean new LOCAL

new: 
	../../Scripts/cdep.py main.c 1

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)


%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@


#################################################################
include dependencies
#################################################################
# Binaries

BIN_FILES             = enkf gen_test.x test.x run_ensemble.x sqlite3_driver_test.x
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

run_ensemble.o	: run_ensemble.c
	$(CC) -c $(CFLAGS) -D'SITE_CONFIG_FILE="$(SITE_CONFIG_FILE)"' -D'SVN_VERSION="$(SVN_VERSION)"' -D'COMPILE_TIME_STAMP="$(COMPILE_TIME_STAMP)"' $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

run_ensemble.x:	run_ensemble.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) run_ensemble.o -o run_ensemble.x  $(LIB_PATH)  $(LIB_LINK) 	

sqlite3_driver_test.x: sqlite3_driver_test.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) sqlite3_driver_test.o -o sqlite3_driver_test.x  $(LIB_PATH)  $(LIB_LINK) 	
