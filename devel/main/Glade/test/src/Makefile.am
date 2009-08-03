## Process this file with automake to produce Makefile.in

INCLUDES = \
	-DPACKAGE_DATA_DIR=\""$(datadir)"\" \
	-DPACKAGE_LOCALE_DIR=\""$(prefix)/$(DATADIRNAME)/locale"\" \
	-I/private/masar/enkf/libutil/include \
	-I/private/masar/enkf/libecl/include \
	-I/private/masar/enkf/librms/include \
	-I/private/masar/enkf/libconfig/include \
	-I/private/masar/enkf/libsched/include \
	-I/private/masar/enkf/libjob_queue/include \
	-I/private/masar/enkf/libanalysis/include \
	-I/private/masar/enkf/libsample/include \
	-I/private/masar/enkf/libenkf/include \
	@PACKAGE_CFLAGS@

bin_PROGRAMS = test

test_SOURCES = \
	main.c \
	support.c support.h \
	interface.c interface.h \
	enkf_setup.c enkf_setup.h \
	callbacks.c callbacks.h \
	custom-cell-renderer-progressbar.c custom-cell-renderer-progressbar.h

test_LDADD = @PACKAGE_LIBS@ $(INTLLIBS) \
			 -L/private/masar/enkf/libutil/lib -lutil \
			 -L/private/masar/enkf/libenkf/lib -lenkf \
			 -L/private/masar/enkf/libecl/lib -lecl \
			 -L/private/masar/enkf/libjob_queue/lib -ljob_queue \
			 -L/private/masar/enkf/libsched/lib -lsched \
			 -L/private/masar/enkf/libconfig/lib -lconfig \
			 -L/private/masar/enkf/libutil/lib -lutil \
			 -L/private/masar/enkf/libanalysis/lib -lanalysis \
			 -L/private/masar/enkf/libsample/lib -lsample \
			 -L/d/proj/bg/enkf/software/lib \
			 -L/d/proj/bg/enkf/software/include \
			 -L/private/masar/enkf/librms/lib -lrms -lm -lpthread -lz  -llapack -lblas -lgfortran -lgtk-x11-2.0 -lgdk-x11-2.0 -latk-1.0 -lgdk_pixbuf-2.0 -lm -lpangoxft-1.0 -lpangox-1.0 -lpango-1.0 -lgobject-2.0 -lgmodule-2.0 -ldl -lgthread-2.0 -lglib-2.0

