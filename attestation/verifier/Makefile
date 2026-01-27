CC ?= gcc
LD ?= ld
LIBPATH ?= /opt/occlum/toolchains/dcap_lib/glibc
INCPATH ?= /opt/occlum/toolchains/dcap_lib/inc

.PHONY: all clean

all: verifier

verifier: verifier.c
		$(CC) $^ -fPIE -pie -o $@ -L $(LIBPATH) -locclum_dcap -I /opt/intel/sgxsdk/include -I $(INCPATH)

clean:
		rm -rf verifier
