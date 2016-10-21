#==========================================================================
# Makefile
#==========================================================================
# @brief: A makefile that compiles and runs the spam filter program
#
# @desc: 1. Enter "make sw" to compile & execute the program on CPU
#        2. Enter "make hls" to run cosimulation
#        3. Enter "make clean" to clean up the directory


# Extract Vivado HLS include path
VHLS_PATH := $(dir $(shell which vivado_hls))/..
VHLS_INC ?= ${VHLS_PATH}/include

VHLS = vivado_hls
HLS_TCL= run.tcl

HEADERS := fileio.hpp \
			defs.h \
			utils.hpp \
			sgd_serial_kernel.h

SOURCES := main_testbench.cpp \
			utils.cpp \
			sgd_serial_kernel.cpp

ALL_DEPS = $(HEADERS) $(SOURCES)

# Specify compilation flags
CFLAGS = -g -I${VHLS_INC} -I../

# Compile & run SW based spam filter implementation
sw: $(ALL_DEPS)
	@echo "Compiling & executing the spam filter program ..."
	g++ ${CFLAGS} $(SOURCES) -o spamfilter
	./spamfilter

# Compile & run HLS based spam filter implementation
hls: $(HLS_TCL) $(ALL_DEPS)
	$(VHLS) $(HLS_TCL)

# Clean up the directory before submission
clean:
	@echo "Clean up output files"
	rm -rf spamfilter vivado_hls.log *.prj result out.dat *~
