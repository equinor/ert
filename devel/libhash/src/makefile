include path_config
MFLAG = m32
CC = gcc
AR = ar
#################################################################
COMPILE_INCLUDE   = -I$(LIBHASH_HOME)/src
INSTALL_INC_PATH  = $(LIBHASH_HOME)/include
INSTALL_LIB_PATH  = $(LIBHASH_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O2 -std=gnu99 -g -Wall -fPIC
ARFLAGS = -csr
#################################################################
OBJECTS   = set.o hash_node.o hash_sll.o hash.o list_node.o list.o node_data.o node_ctype.o
INC_FILES = set.h hash.h hash_node.h hash_sll.h list_node.h list.h node_data.h node_ctype.h
LIB       = libhash_$(MFLAG).a 
LIB_LINK  = -lhash_$(MFLAG) -lz -lm -lpthread



all: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) -I$(INSTALL_INC_PATH) hash_test.c -o hash.x  -L$(INSTALL_LIB_PATH) $(LIB_LINK)

clean:
	rm -f *.o *~ 
	rm -f libhash_*
	rm -f $(INSTALL_INC_PATH)/*.h
	rm -f $(INSTALL_LIB_PATH)/$(LIB)

$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE)  $(CPPFLAGS) $< -o $@


node_ctype.o : node_ctype.h node_ctype.c
set.o        : set.c set.h hash.o
node_data.o  : node_data.c node_data.h
list_node.o  : list_node.c list_node.h
list.o       : list.c list.h list_node.o node_data.o
hash_node.o  : hash_node.c hash_node.h
hash_sll.o   : hash_sll.c hash_sll.h hash_node.o
hash.o       : hash.c hash.h hash_sll.o node_data.o


