include global_config
SDP_ROOT          = $(shell get_sdp_root.py)
#################################################################
COMPILE_INCLUDE  = -I$(LIBSCHED_HOME)/src -I$(LIBUTIL_HOME)/include -I$(LIBECL_HOME)/include 
INSTALL_INC_PATH =   $(LIBSCHED_HOME)/include
INSTALL_LIB_PATH =   $(LIBSCHED_HOME)/lib
#################################################################
OBJECTS   = sched_kw.o sched_file.o  sched_kw_untyped.o sched_kw_gruptree.o sched_kw_tstep.o sched_kw_dates.o sched_kw_wconhist.o sched_kw_wconinjh.o sched_kw_welspecs.o sched_util.o gruptree.o history.o sched_kw_wconprod.o sched_kw_wconinj.o sched_kw_wconinje.o sched_kw_compdat.o sched_kw_include.o
INC_FILES = sched_file.h sched_kw.h sched_kw_untyped.h sched_kw_gruptree.h sched_kw_tstep.h sched_kw_dates.h sched_kw_wconhist.h sched_kw_wconinjh.h sched_kw_welspecs.h sched_util.h gruptree.h history.h sched_kw_wconprod.h sched_kw_wconinj.h  sched_kw_wconinje.h sched_kw_compdat.h sched_kw_include.h sched_macros.h
LIB       = libsched.a

LIB:	$(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)


clean:
	rm -f *.o *~ 
	rm -f libsched.a
	rm -f $(INSTALL_LIB_PATH)/libsched.a
	rm -f $(INSTALL_INC_PATH)/*.h


SDP_INSTALL: LIB 
	install $(LIB) $(SDP_ROOT)/lib/$(LIB_ROOT).a
	install $(INC_FILES) $(SDP_ROOT)/include

rebuild: clean all


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $< -o $@


new: 
	../../Scripts/cdep.py all_include.h 0

include dependencies

