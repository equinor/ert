include path_config
MFLAG = m32
CC = gcc
AR = ar
FF = ifort
#################################################################
COMPILE_INCLUDE   = -I./  -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include
COMPILE_LIB_PATH  = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib 
INSTALL_INC_PATH = $(LIBKERNEL_HOME)/include
INSTALL_LIB_PATH = $(LIBKERNEL_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -std=gnu99 -g -Wall -fPIC -O3
FFLAGS  = -implicitnone -nozero -warn all -c -O3 -integer-size 64 -real_size 64 -assume byterecl -assume nobuffered_io -convert big_endian
ARFLAGS = -csr
#################################################################
OBJECTS           =  kernel_core.o cost_func.o pre-image.o
INC_FILES 				= kernel_core.h cost_func.h pre-image.h
LIB = libkernel_$(MFLAG).a
LIB_LINK          = -lsample_m32 -lutil_$(MFLAG) -lhash_$(MFLAG) -lz -lm -lbfgs -lblas -lifcore -lguide -lfftw3 -llapack -lanalysis_m32 -lirc
LINK_LIB_PATH     = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L/local/intel/compiler90/ia32/fc/lib -L/h/jgv/lib -L/h/a152128/lib/linux
FINAL_INC_PATH    = -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include

all: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

clean:
	rm -f *.o
	rm -f *,mod
	rm -f $(LIB)
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/$(LIB)


%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

%.o : %.f90
	$(FF) -c $(FFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@
