include path_config
MFLAG = m32
CC = gcc
AR = ar
FF = ifort
#################################################################
COMPILE_INCLUDE   = -I./  -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBSAMPLE_HOME)/include 
COMPILE_LIB_PATH  = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L/h/jgv/lib
INSTALL_INC_PATH  = $(LIBKERNEL_HOME)/include
INSTALL_LIB_PATH  = $(LIBKERNEL_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -Wall -fPIC -std=gnu99 -g
FFLAGS  = -O2 -openmp -fpe0 -traceback -warn all -CB -u -g -real_size 64 -assume byterecl -assume nobuffered_io -convert big_endian
ARFLAGS = -csr
#################################################################
OBJECTS           = kernel_core.o cost_func.o pre-image.o
INC_FILES 				= kernel_core.h cost_func.h pre-image.h
LIB               = libkernel_$(MFLAG).a
LIB_LINK          = -lsample_$(MFLAG) -lanalysis_$(MFLAG) -lkernel_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG)  -lz -lg2c -lmkl_lapack -lmkl -lifcoremt -lguide -lfftw3_$(MFLAG)  -lbfgs 
LIB_PATH          = -L/usr/lib/gcc/x86_64-redhat-linux/3.4.3/32  -L$(LIBSAMPLE_HOME)/lib -L$(LIBANALYSIS_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBBFGS_HOME)/lib -L$(LIBKERNEL_HOME)/lib -L$(EXTERNAL_LIB_PATH) -L/local/intel/compiler90/ia32/fc/lib -L/local/intel/mkl/mkl721/lib/32
TARGET_INCLUDE    = -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBKERNEL_HOME)/include

all: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	$(CC) $(CFLAGS) $(TARGET_INCLUDE) -c test_kernel.c gnuplot.c  
	$(CC) $(CFLAGS) test_kernel.o gnuplot.o -o test_kernel.x $(LIB_PATH) $(LIB_LINK)
	$(FF) $(FFLAGS) -c test_fortran.f90
	$(FF) test_fortran.o -o test_fortran.x $(LIB_PATH) $(LIB_LINK)
	$(FF) $(FFLAGS) -c test_fortran_omp.f90
	$(FF) test_fortran_omp.o -o test_fortran_omp.x $(LIB_PATH) $(LIB_LINK)

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

clean:
	rm -f *.o
	rm -f *,mod
	rm -f $(LIB)
	rm -f *.plt


%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

%.o : %.f
	$(FF) -c $(FFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@
