include global_config
SDP_ROOT          = $(shell get_sdp_root.py)
#################################################################
COMPILE_INCLUDE  = -I$(LIBSCHED_HOME)/src -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include 
INSTALL_INC_PATH =   $(LIBSCHED_HOME)/include
INSTALL_LIB_PATH =   $(LIBSCHED_HOME)/lib
#################################################################
OBJECTS   = sched_kw.o sched_file.o  sched_kw_untyped.o sched_kw_gruptree.o sched_kw_tstep.o sched_kw_dates.o sched_kw_wconhist.o sched_kw_wconinjh.o sched_kw_welspecs.o sched_util.o gruptree.o history.o sched_kw_wconprod.o sched_kw_wconinj.o sched_kw_wconinje.o
INC_FILES = sched_file.h sched_kw.h sched_kw_untyped.h sched_kw_gruptree.h sched_kw_tstep.h sched_kw_dates.h sched_kw_wconhist.h sched_kw_wconinjh.h sched_kw_welspecs.h sched_util.h gruptree.h history.h sched_kw_wconprod.h sched_kw_wconinj.h  sched_kw_wconinje.h
LIB       = libsched_$(MFLAG).a
LIB_LINK  = -lsched_$(MFLAG) -lecl_$(MFLAG) -lutil_$(MFLAG) -lm -lz -lpt
LIB_PATH  = -L$(LIBUTIL_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib
TARGET_INCLUDE = -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBSCHED_HOME)/include

all: LIB
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

clean:
	rm -f *.o *~ 
	rm -f libsched_$(MFLAG).a
	rm -f $(INSTALL_LIB_PATH)/libsched_$(MFLAG).a
	rm -f $(INSTALL_INC_PATH)/*.h


SDP_INSTALL: LIB BIN
	install $(LIB) $(SDP_ROOT)/lib/$(LIB_ROOT).a
	install $(BIN_FILES_SDP_INSTALL) $(SDP_ROOT)/bin
	install $(INC_FILES) $(SDP_ROOT)/include

rebuild: clean all

LIB:	$(LIB)

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@



#################################################################
# Binaries
# Observe that the binaries are built against the libraries / headers



BIN_FILES = sched_test.x gruptree_test.x sched_summary.x
BIN	: $(BIN_FILES)
BIN_FILES_SDP_INSTALL = sched_summary.x


sched_test.x: $(LIB) sched_test.o
	$(LD) -$(MFLAG) $(LDFLAGS) sched_test.o -o sched_test.x $(LIB_PATH) $(LIB_LINK)

gruptree_test.x: $(LIB) gruptree_test.o
	$(LD) -$(MFLAG) $(LDFLAGS) gruptree_test.o -o gruptree_test.x $(LIB_PATH) $(LIB_LINK)

sched_summary.x: $(LIB) sched_summary.o
	$(LD) -$(MFLAG) $(LDFLAGS) sched_summary.o -o sched_summary.x $(LIB_PATH) $(LIB_LINK)
