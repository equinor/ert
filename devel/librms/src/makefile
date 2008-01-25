include path_config
MFLAG = m32
CC = gcc
AR = ar
#################################################################
COMPILE_INCLUDE   = -I$(LIBRMS_HOME)/src -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include
INSTALL_INC_PATH  = $(LIBRMS_HOME)/include
INSTALL_LIB_PATH  = $(LIBRMS_HOME)/lib
#################################################################
CFLAGS  = -$(MFLAG) -O2 -std=gnu99 -g -Wall 
ARFLAGS = -csr
#################################################################
OBJECTS           = rms_interC.o rms_file.o rms_util.o rms_tag.o rms_type.o rms_tagkey.o rms_stats.o
INC_FILES_INSTALL = rms_file.h rms_util.h rms_tag.h rms_type.h rms_tagkey.h rms_stats.h
LIB               = librms_$(MFLAG).a
LIB_LINK          = -lrms_$(MFLAG) -lecl_$(MFLAG) -lhash_$(MFLAG) -lutil_$(MFLAG) -lm -lz
LIB_PATH          = -L$(LIBUTIL_HOME)/lib -L$(LIBHASH_HOME)/lib -L$(LIBECL_HOME)/lib -L$(LIBRMS_HOME)/lib
TARGET_INCLUDE    = -I$(LIBECL_HOME)/include -I$(LIBUTIL_HOME)/include -I$(LIBHASH_HOME)/include -I$(LIBRMS_HOME)/include

INSTALL: $(LIB)
	install -d $(INSTALL_INC_PATH)
	install -d $(INSTALL_LIB_PATH)
	install $(INC_FILES_INSTALL) $(INSTALL_INC_PATH)
	install $(LIB) $(INSTALL_LIB_PATH)
	gcc $(CFLAGS) $(TARGET_INCLUDE)	rms_test.c    	-o rms.x               $(LIB_PATH)  $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE)	rms_stat.c    	-o rms_stat.x          $(LIB_PATH)  $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE)	tag_list.c    	-o tag_list.x          $(LIB_PATH)  $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE)	rms_extract.c 	-o rms_extract.x       $(LIB_PATH)  $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE)	rms_setname.c 	-o rms_setname.x       $(LIB_PATH)  $(LIB_LINK) 
	gcc $(CFLAGS) $(TARGET_INCLUDE)	condor_import.c -o /d/proj/bg/enkf/bin/condor_import.x   $(LIB_PATH)  $(LIB_LINK) 

clean:
	rm -f *.o *~ 
	rm -f $(LIB)
	rm -f $(INSTALL_LIB_PATH)/$(LIB)
	rm -f $(INSTALL_INC_PATH)/*.h


$(LIB): $(OBJECTS)
	$(AR) $(ARFLAGS) $(LIB) $(OBJECTS)

%.o : %.c
	$(CC) -c $(CFLAGS) $(COMPILE_INCLUDE) $(CPPFLAGS) $< -o $@



rms_stats.o  : rms_stats.c rms_stats.h
rms_type.o   : rms_type.c rms_type.h
rms_util.o   : rms_util.c rms_util.h
rms_tagkey.o : rms_tagkey.c rms_tagkey.h rms_type.o rms_util.o
rms_tag.o    : rms_tag.c rms_tag.h rms_tagkey.o rms_util.o
rms_file.o   : rms_file.c rms_file.h rms_util.o rms_tag.o rms_tagkey.o
rms_interC.o : rms_interC.c rms_file.o

