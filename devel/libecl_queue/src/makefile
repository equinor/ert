include path_config
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
OBJECTS   = ecl_queue.o basic_queue_driver.o lsf_driver.o
INC_FILES = ecl_queue.h basic_queue_driver.h
LIB       = libecl_queue_$(MFLAG).a
LIB_PATH  = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_QUEUE_HOME)/lib
LIB_LINK  = -lecl_$(MFLAG) -lhash_$(MFLAG) -lutil_$(MFLAG) -lm -lz
TARGET_INCLUDE  = -I$(LIBECL_QUEUE_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include

INSTALL: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

clean:
	rm -f *.o *~ 
	rm -f libecl_*
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/*.a


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@


ecl_queue.o          : ecl_queue.c ecl_queue.h basic_queue_driver.o
basic_queue_driver.o : basic_queue_driver.c basic_queue_driver.h
