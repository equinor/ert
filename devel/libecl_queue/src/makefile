include path_config
#m32 -> #define i386 1    m64 -> __x86_64 1
MFLAG = m32
CC = gcc
AR = ar
#################################################################
COMPILE_INCLUDE  = -I$(LIBECL_QUEUE_HOME)/src -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBECL_HOME)/include -I$(EXTERNAL_INC_PATH)
INSTALL_INC_PATH =  $(LIBECL_QUEUE_HOME)/include
INSTALL_LIB_PATH =  $(LIBECL_QUEUE_HOME)/lib
#################################################################
CFLAGS  = -O2 -$(MFLAG)  -std=gnu99 -g -Wall -fPIC
ARFLAGS = -csr
#################################################################
OBJECTS   = job_queue.o basic_queue_driver.o lsf_driver.o local_driver.o rsh_driver.o ext_job.o ext_joblist.o
INC_FILES = job_queue.h basic_queue_driver.h lsf_driver.h local_driver.h rsh_driver.h ext_job.h ext_joblist.h
LIB       = libecl_queue_$(MFLAG).a
LIB_PATH  = -L$(EXTERNAL_LIB_PATH) -L$(LIBECL_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_QUEUE_HOME)/lib
LIB_LINK  = -lecl_queue_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lm -lz -lbat  -llsf -lnsl 
TARGET_INCLUDE  = -I$(LIBECL_QUEUE_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include

INSTALL: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	#gcc $(CFLAGS) $(TARGET_INCLUDE) ecl_queue_test.c         -o queue.x  $(LIB_PATH)  $(LIB_LINK)
	gcc $(CFLAGS) $(TARGET_INCLUDE) ext_test.c                -o ext.x  $(LIB_PATH)  $(LIB_LINK)

clean:
	rm -f *.o *~ 
	rm -f libecl_*
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/$(LIB)


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@


job_queue.o          : job_queue.c job_queue.h basic_queue_driver.o
basic_queue_driver.o : basic_queue_driver.c basic_queue_driver.h
lsf_driver.o         : lsf_driver.c lsf_driver.h
local_driver.o       : local_driver.c local_driver.h
rsh_driver.o         : rsh_driver.c rsh_driver.h
ext_job.o            : ext_job.c ext_job.h
ext_joblist.o        : ext_joblist.c ext_joblist.h ext_job.o

