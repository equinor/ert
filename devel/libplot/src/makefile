include path_config
MFLAG = m64
CC = gcc
AR = ar
SDP_ROOT          = /project/res/$(shell /private/joaho/bin/res_target.py)
#################################################################
COMPILE_INCLUDE   = -I$(LIBPLOT_HOME)/src -I/h/masar/plplot-5.9.0/Compilation/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBECL_HOME)/include
INSTALL_INC_PATH  = $(LIBPLOT_HOME)/include
INSTALL_LIB_PATH  = $(LIBPLOT_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O2 -std=gnu99 -g -Wall -fPIC
ARFLAGS = -csr
#################################################################
OBJECTS       = plot.o plot_item.o
INC_FILES     = plot.h
LIB_ROOT      = plot
LIB           = lib$(LIB_ROOT)_$(MFLAG).a 
LIB_LINK      = -l$(LIB_ROOT)_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lecl_$(MFLAG) -lz -lm -lplot_$(MFLAG) -lplplotd 
LIB_LINK_PATH = -L./ -L/h/masar/plplot-5.9.0/Compilation/lib -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib 

LOCAL: LIB BIN
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

SDP_INSTALL: LIB BIN
	install $(LIB) $(SDP_ROOT)/lib/lib$(LIB_ROOT).a
	install $(BIN_FILES) $(SDP_ROOT)/bin
	install $(INC_FILES) $(SDP_ROOT)/include

LIB:	$(LIB)


clean:
	rm -f *.o *~ *.x
	rm -f libplot_*
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/$(LIB)

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE)  $(CPPFLAGS) $< -o $@

# Library source code
plot.o : plot.h plot.c plot_item.c


#Binaries
plot_test.o : plot_test.c

BIN:	    plot.x summary_plot.x
BIN_FILES = plot.x summary_plot.x


plot.x: plot_test.o $(LIB)
	gcc $(LDFLAGS) plot_test.o -o plot.x     $(LIB_LINK_PATH) $(LIB_LINK) 

summary_plot.x: summary_plot.o $(LIB)
	gcc $(LDFLAGS) summary_plot.o -o summary_plot.x     $(LIB_LINK_PATH) $(LIB_LINK) 
