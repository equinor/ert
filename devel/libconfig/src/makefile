include global_config
SDP_ROOT          = $(shell get_sdp_root.py)
#################################################################
COMPILE_INCLUDE   = -I./  -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include
COMPILE_LIB_PATH  = -L$(LIBHASH_HOME)/lib -L$(LIBUTIL_HOME)/lib

INSTALL_INC_PATH = $(LIBCONFIG_HOME)/include
INSTALL_LIB_PATH = $(LIBCONFIG_HOME)/lib
#################################################################
OBJECTS           = config.o conf.o conf_util.o conf_data.o
INC_FILES_INSTALL = config.h conf.h conf_data.h
LIB               = libconfig.a
LIB_LINK          = -lconfig -lutil -lz -lpthread -lm
LINK_LIB_PATH     = -L./ -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib

LOCAL: LIB
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

SDP_INSTALL: LIB
	install $(LIB) $(SDP_ROOT)/lib/$(LIB_ROOT).a
	install $(INC_FILES_INSTALL) $(SDP_ROOT)/include

clean:
	rm -f *.o *~ 
	rm -f libconfig_*
	rm -f $(INSTALL_LIB_PATH)/$(LIB)
	rm -f $(INSTALL_INC_PATH)/*.h

LIB:	$(LIB)


rebuild: clean LOCAL	

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

new: 
	../../Scripts/cdep.py all_include.h 0

include dependencies

