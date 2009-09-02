include global_config
#################################################################
COMPILE_INCLUDE  = -I$(LIBJOB_QUEUE_HOME)/src -I$(LIBCONFIG_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBECL_HOME)/include -I$(EXTERNAL_INC_PATH)
INSTALL_INC_PATH =  $(LIBJOB_QUEUE_HOME)/include
INSTALL_LIB_PATH =  $(LIBJOB_QUEUE_HOME)/lib
SDP_ROOT 	 = $(shell get_sdp_root.py)
#################################################################
OBJECTS   = forward_model.o job_queue.o basic_queue_driver.o lsf_driver.o local_driver.o rsh_driver.o ext_job.o ext_joblist.o lsf_request.o
INC_FILES = job_queue.h basic_queue_driver.h lsf_driver.h local_driver.h rsh_driver.h ext_job.h ext_joblist.h lsf_request.h forward_model.h 
LIB       = libjob_queue.a
LIB_PATH  = -L$(EXTERNAL_LIB_PATH) -L$(LIBCONFIG_HOME)/lib -L$(LIBUTIL_HOME)/lib -L./
LIB_LINK  = -ljob_queue -lconfig -lutil -lm -lz -lbat  -llsf -lnsl -lpthread

LOCAL: LIB 
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)


SDP_INSTALL: LIB BIN
	install $(LIB) $(SDP_ROOT)/lib/$(LIB_ROOT).a
	#install $(BIN_FILES_SDP_INSTALL) $(SDP_ROOT)/bin
	install $(INC_FILES) $(SDP_ROOT)/include

clean:
	rm -f *.o *~ 
	rm -f libjob_*
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/$(LIB)

LIB:	$(LIB)

rebuild: clean LOCAL

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)


ifeq ($(USE_LSF_LIBRARY),TRUE)
   CFLAGS_LSF = $(CFLAGS) -D'USE_LSF_LIBRARY=TRUE'
else
   ifeq ($(xStatoil),TRUE)
      CFLAGS_LSF = $(CFLAGS) -D'xStatoil=TRUE'
   else
      CFLAGS_LSF = $(CFLAGS) 
   endif
endif

lsf_driver.o : lsf_driver.c
	$(CC) -c $(CFLAGS_LSF) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

new: 
	../../Scripts/cdep.py all_include.h 0

include dependencies


#BIN_FILES
#BIN	:  $(BIN_FILES) 



