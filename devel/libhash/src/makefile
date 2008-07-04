include path_config
MFLAG = m64
CC = gcc
AR = ar
SDP_ROOT          = /project/res/$(shell /private/joaho/bin/res_target.py)
#################################################################
COMPILE_INCLUDE   = -I$(LIBHASH_HOME)/src
INSTALL_INC_PATH  = $(LIBHASH_HOME)/include
INSTALL_LIB_PATH  = $(LIBHASH_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O2 -std=gnu99 -g -Wall -fPIC
ARFLAGS = -csr
#################################################################
OBJECTS       = set.o hash_node.o hash_sll.o hash.o list_node.o list.o node_data.o node_ctype.o
INC_FILES     = set.h hash.h hash_node.h hash_sll.h list_node.h list.h node_data.h node_ctype.h
LIB_ROOT      = hash
LIB           = lib$(LIB_ROOT)_$(MFLAG).a 
LIB_LINK      = -l$(LIB_ROOT)_$(MFLAG) -lz -lm -lpthread
LIB_LINK_PATH = -L./

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
	rm -f *.o *~ 
	rm -f libhash_*
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/$(LIB)

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE)  $(CPPFLAGS) $< -o $@

# Library source code
node_ctype.o : node_ctype.h node_ctype.c
set.o        : set.c set.h hash.o
node_data.o  : node_data.c node_data.h
list_node.o  : list_node.c list_node.h
list.o       : list.c list.h list_node.o node_data.o
hash_node.o  : hash_node.c hash_node.h
hash_sll.o   : hash_sll.c hash_sll.h hash_node.o
hash.o       : hash.c hash.h hash_sll.o node_data.o


#Binaries
hash_test.o : hash_test.c

BIN:	    hash.x
BIN_FILES = hash.x


hash.x: hash_test.o $(LIB)
	gcc $(LDFLAGS) -$(MFLAG) hash_test.o -o hash.x     $(LIB_LINK_PATH) $(LIB_LINK) 
