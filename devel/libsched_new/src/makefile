include path_config
MFLAG = m64
CC = gcc
AR = ar



#################################################################
COMPILE_INCLUDE  = -I$(LIBSCHED_HOME)/src -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include
INSTALL_INC_PATH =   $(LIBSCHED_HOME)/include
INSTALL_LIB_PATH =   $(LIBSCHED_HOME)/lib
#################################################################
CFLAGS  = -O2 -$(MFLAG) -std=gnu99 -g -Wall 
FFLAGS  = -fpe0 -traceback -warn unused -CB -u -g -real_size 64 -assume byterecl -assume nobuffered_io -convert big_endian
ARFLAGS = -csr
#################################################################
OBJECTS   = sched_file.o sched_kw.o sched_kw_untyped.o sched_kw_gruptree.o sched_kw_tstep.o sched_kw_dates.o sched_kw_wconhist.o sched_kw_welspecs.o sched_util.o gruptree.o history.o
INC_FILES = sched_file.h sched_kw.h sched_kw_untyped.h sched_kw_gruptree.h sched_kw_tstep.h sched_kw_dates.h sched_kw_wconhist.h sched_kw_welspecs.h sched_util.h gruptree.h history.h
LIB       = libsched_$(MFLAG).a
LIB_LINK  = -lsched_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG)  -lm -lz
LIB_PATH  = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib
TARGET_INCLUDE = -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBSCHED_HOME)/include

all: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) $(TARGET_INCLUDE) sched_test.c -o sched_test.x $(LIB_PATH) $(LIB_LINK)
	gcc $(CFLAGS) $(TARGET_INCLUDE) gruptree_test.c -o gruptree_test.x $(LIB_PATH) $(LIB_LINK)


clean:
	rm -f *.o *~ 
	rm -f libsched_$(MFLAG).a
	rm -f $(INSTALL_LIB_PATH)/libsched_$(MFLAG).a
	rm -f $(INSTALL_INC_PATH)/*.h



$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@

%.o : %.f90
	$(FC) -c $(FFLAGS) $< -o $@
