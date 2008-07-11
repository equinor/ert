include path_config
MFLAG = m64
CC = gcc
AR = ar
#################################################################
COMPILE_INCLUDE  = -I$(LIBTPG_HOME)/src -I$(LIBENKF_HOME)/include -I$(LIBRMS_HOME)/include -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include
INSTALL_INC_PATH =   $(LIBTPG_HOME)/include
INSTALL_LIB_PATH =   $(LIBTPG_HOME)/lib
#################################################################
CFLAGS  = -O2 -$(MFLAG) -std=gnu99 -g -Wall 
FFLAGS  = -fpe0 -traceback -warn unused -CB -u -g -real_size 64 -assume byterecl -assume nobuffered_io -convert big_endian
ARFLAGS = -csr
#################################################################
OBJECTS   = tpgzone.o trs.o petp.o tpgzone_util.o
INC_FILES = tpgzone.h trs.h petp.h 
LIB       = libtpg_$(MFLAG).a
LIB_LINK  = -ltpg_$(MFLAG) -lenkf_$(MFLAG) -lrms_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG)  -lm -lz
LIB_PATH  = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBRMS_HOME)/lib -L$(LIBENKF_HOME)/lib -L$(LIBTPG_HOME)/lib
TARGET_INCLUDE = -I$(LIBTPG_HOME)/include -I$(LIBENKF_HOME)/include -I$(LIBECL_HOME)/include -I$(LIBRMS_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include 

all: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) $(TARGET_INCLUDE) tpgzone_test.c -o tpgzone_test.x $(LIB_PATH) $(LIB_LINK)



clean:
	rm -f *.o *~ 
	rm -f libtpg_$(MFLAG).a
	rm -f $(INSTALL_LIB_PATH)/libtpg_$(MFLAG).a
	rm -f $(INSTALL_INC_PATH)/*.h



$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@

%.o : %.f90
	$(FC) -c $(FFLAGS) $< -o $@


tpgzone.o       : tpgzone.h trs.h petp.h tpgzone_util.h
petp.o          : petp.h tpgzone_util.h
trs.o           : trs.h
tpgzone_util.o  : tpgzone_util.h
