include path_config

FF = ifort

FFLAGS = -fpe0 -traceback -warn all -CB -u -g -c -real_size 64 -assume byterecl -assume nobuffered_io -convert big_endian

all: libbfgs.a
	install -d $(LIBBFGS_HOME)/lib
	install libbfgs.a $(LIBBFGS_HOME)/lib

libbfgs.a: bfgs.o
	ar csr libbfgs.a bfgs.o

bfgs.o: bfgs.f
	$(FF) $(FFLAGS) -o bfgs.o bfgs.f

clean:
	rm -r *.o *.a
	rm $(LIBBFGS_HOME)/lib/libbfgs.a
