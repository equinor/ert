include global_config
SDP_ROOT	 = $(shell get_sdp_root.py)
COMPILE_INCLUDE  = -I./ -I$(LIBECL_HOME)/src -I$(LIBUTIL_HOME)/include -I$(LIBCONFIG_HOME)/include -I$(LIBSCHED_HOME)/include -I$(EXTERNAL_INC_PATH) 

LIB_PATH  = -L$(LIBUTIL_HOME)/lib  -L$(LIBECL_HOME)/lib -L$(LIBSCHED_HOME)/lib -L$(LIBCONFIG_HOME)/lib
LIB_LINK  = -lsched -lecl -lconfig -lutil  -lm -lz -lpthread -llapack


BIN:	APPLICATIONS


APPLICATION_LIST  = perturb_history
APPLICATIONS: $(APPLICATION_LIST)

OBJECTS = perturb_history.o pert_util.o well_rate.o group_rate.o

clean:
	rm -f *.o
	rm -f $(APPLICATION_LIST)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

perturb_history.o : perturb_history.c

#################################################################
# Binaries
# Observe that the binaries are built against the libraries / headers
# in the source directory - and not the installed files. This is to



perturb_history:	$(OBJECTS)
	$(LD) -$(MFLAG) $(LDFLAGS)  $(OBJECTS) -o perturb_history $(LIB_PATH) -L$(EXTERNAL_LIB_PATH) $(LIB_LINK) 


