.PHONY: all clean

CC = hipcc
CFLAGS = --offload-arch=gfx908 -fPIC
INC = -Iinc/
HIP_LIB = /opt/rocm-6.0.2/lib

OBJS = saxpy.o sscal.o sswap.o scopy.o sdot.o sdsdot.o sasum.o snrm2.o isamax.o libchatblas.so 
#OBJS = saxpy.o sscal.o sswap.o scopy.o sdot.o sasum.o snrm2.o isamax.o libchatblas.so 

all: $(OBJS)

saxpy.o: saxpy.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

sscal.o: sscal.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

sswap.o: sswap.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

scopy.o: scopy.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

sdot.o: sdot.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

sdsdot.o: sdsdot.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

sasum.o: sasum.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

snrm2.o: snrm2.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

isamax.o: isamax.cpp
	$(CC) -c $(CFLAGS) $(INC) $< -o $@

libchatblas.so: 
	gcc -L$(HIP_LIB) -lamdhip64 -o $@ saxpy.o sscal.o sswap.o scopy.o sdot.o sdsdot.o sasum.o snrm2.o isamax.o -shared
	#gcc -L$(HIP_LIB) -lamdhip64 -o $@ saxpy.o sscal.o sswap.o scopy.o sdot.o sasum.o snrm2.o isamax.o -shared

clean:
	rm -f $(OBJS)
