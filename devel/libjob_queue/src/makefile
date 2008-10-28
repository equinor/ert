include global_config
#################################################################
COMPILE_INCLUDE  = -I$(LIBJOB_QUEUE_HOME)/src -I$(LIBCONFIG_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBECL_HOME)/include -I$(EXTERNAL_INC_PATH)
INSTALL_INC_PATH =  $(LIBJOB_QUEUE_HOME)/include
INSTALL_LIB_PATH =  $(LIBJOB_QUEUE_HOME)/lib
SDP_ROOT         = /project/res/$(shell /private/joaho/bin/res_target.py)
#################################################################
OBJECTS   = job_queue.o basic_queue_driver.o lsf_driver.o local_driver.o rsh_driver.o ext_job.o ext_joblist.o lsf_request.o
INC_FILES = job_queue.h basic_queue_driver.h lsf_driver.h local_driver.h rsh_driver.h ext_job.h ext_joblist.h lsf_request.h
LIB_ROOT  = job_queue
LIB       = lib$(LIB_ROOT)_$(MFLAG).a
LIB_PATH  = -L$(EXTERNAL_LIB_PATH) -L$(LIBCONFIG_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L./
LIB_LINK  = -l$(LIB_ROOT)_$(MFLAG) -lconfig_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lm -lz -lbat  -llsf -lnsl 

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

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)


ifeq ($(USE_LSF_LIBRARY),TRUE)
   CFLAGS_LSF = $(CFLAGS) -D'USE_LSF_LIBRARY=TRUE'
else
   CFLAGS_LSF = $(CFLAGS) 
endif

lsf_driver.o : lsf_driver.c
	$(CC) -c $(CFLAGS_LSF) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

#Library
job_queue.o          : job_queue.c job_queue.h basic_queue_driver.o
basic_queue_driver.o : basic_queue_driver.c basic_queue_driver.h
lsf_driver.o         : lsf_driver.c lsf_driver.h
local_driver.o       : local_driver.c local_driver.h
rsh_driver.o         : rsh_driver.c rsh_driver.h
ext_job.o            : ext_job.c ext_job.h 
ext_joblist.o        : ext_joblist.c ext_joblist.h ext_job.o
lsf_request.o        : lsf_request.c lsf_request.h ext_job.o ext_joblist.o job_queue.o

BIN_FILES  = test.x
BIN	:  $(BIN_FILES) 

test.x: $(LIB) test.o
	$(LD) -$(MFLAG) $(LDFLAGS) test.o -o test.x $(LIB_PATH) $(LIB_LINK)



