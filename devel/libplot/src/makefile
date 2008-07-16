include path_config
MFLAG = m64
CC = gcc
AR = ar
#################### PLPLOT LINKING #############################
LIBPLPLOT_HOME = /h/masar/plplot-5.9.0/Compilation
LIBPLPLOT_LINK = plplotd
#################################################################
COMPILE_INCLUDE   = -I$(LIBPLOT_HOME)/src -I$(LIBPLPLOT_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBECL_HOME)/include 
INSTALL_INC_PATH  = $(LIBPLOT_HOME)/include
INSTALL_LIB_PATH  = $(LIBPLOT_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O2 -std=gnu99 -g -Wall -fPIC
ARFLAGS = -csr
#################################################################
OBJECTS       = plot.o plot_dataset.o
INC_FILES     = plot.h plot_dataset.h
LIB           = libplot_$(MFLAG).a 
LIB_LINK      = -lplot_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lecl_$(MFLAG) -lz -lm -l$(LIBPLPLOT_LINK) -lpthread 
LIB_PATH = -L$(LIBPLOT_HOME)/lib -L$(LIBPLPLOT_HOME)/lib -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib 
TARGET_INCLUDE   = -I$(LIBPLOT_HOME)/src -I$(LIBPLPLOT_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBECL_HOME)/include


all: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) $(TARGET_INCLUDE) plot_test.c -o plot.x $(LIB_PATH) $(LIB_LINK)
	gcc $(CFLAGS) $(TARGET_INCLUDE) summary_plot.c -o summary_plot.x $(LIB_PATH) $(LIB_LINK)
	gcc $(CFLAGS) $(TARGET_INCLUDE) simple.c -o simple.x $(LIB_PATH) $(LIB_LINK)

clean:
	rm -f *.o *~ *.x
	rm -f libplot_$(MFLAG).a
	rm -f $(INSTALL_LIB_PATH)/libplot_$(MFLAG).a
	rm -f $(INSTALL_INC_PATH)/*.h


distclean:
	rm -f *.o *~ *.x
	rm -f libplot_$(MFLAG).a
	rm -f $(INSTALL_LIB_PATH)/libplot_$(MFLAG).a
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -rf Doc/html Doc/latex

doc:
	doxygen Doc/libplot.conf


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@

plot.o          : plot.h
plot_dataset.o  : plot_dataset.h
