include global_config
#################################################################
COMPILE_INCLUDE   = -I./  -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include
COMPILE_LIB_PATH  = -L$(LIBHASH_HOME)/lib -L$(LIBUTIL_HOME)/lib

INSTALL_INC_PATH = $(LIBCONFIG_HOME)/include
INSTALL_LIB_PATH = $(LIBCONFIG_HOME)/lib
#################################################################
SDP_ROOT          = /project/res/$(shell /private/joaho/bin/res_target.py)
OBJECTS           = config.o cfg.o cfg_lang.o
INC_FILES_INSTALL = config.h
LIB_ROOT          = libconfig
LIB               = $(LIB_ROOT)_$(MFLAG).a
LIB_LINK          = -lconfig_$(MFLAG) -lutil_$(MFLAG) -lhash_$(MFLAG) -lz -lpthread -lm
LINK_LIB_PATH     = -L./ -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib

LOCAL: LIB BIN
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)

SDP_INSTALL: LIB BIN
	install $(LIB) $(SDP_ROOT)/lib/$(LIB_ROOT).a
	install $(BIN_FILES) $(SDP_ROOT)/bin
	install $(INC_FILES_INSTALL) $(SDP_ROOT)/include

clean:
	rm -f *.o *~ 
	rm -f libconfig_*
	rm -f $(INSTALL_LIB_PATH)/$(LIB)
	rm -f $(INSTALL_INC_PATH)/*.h

LIB:	$(LIB)


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@

#Library source code:
config.o      : config.c config.h 
cfg.o         : cfg.c
cfg_lang.o    : cfg_lang.c

# Binaries

#################################################################

BIN: config_test.x
BIN_FILES = config_test.x
 
config_test.x: config_test.o $(LIB)
	$(LD) -$(MFLAG) $(LDFLAGS) config_test.o -o config_test.x     $(LINK_LIB_PATH) $(LIB_LINK) 

