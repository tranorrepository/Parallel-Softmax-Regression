SDACCEL := sdaccel

SW_EMUL_TCL := scripts/sw_emulation.tcl
HW_EMUL_TCL := scripts/hw_emulation.tcl
DEBUG_TCL   := scripts/debug.tcl
BOARD_COMPILE_TCL := scripts/board_compile.tcl
BOARD_RUN_TCL := scripts/board_run.tcl

HARNESS_DIR = harness/sdaccel/
UTILS_DIR = src/

HEADERS := $(HARNESS_DIR)/CLKernel.h \
			$(HARNESS_DIR)/CLMemObj.h \
			src/SdaccelSgd.h \
			$(UTILS_DIR)/fileio.hpp \
			$(UTILS_DIR)/defs.h \
			$(UTILS_DIR)/utils.hpp

SOURCES := src/main_sdaccel.cpp \
			$(HARNESS_DIR)/CLKernel.cpp \
			$(HARNESS_DIR)/CLMemObj.cpp \
			src/SdaccelSgd.cpp \
			$(UTILS_DIR)/utils.cpp

sw_emulation: $(SW_EMUL_TCL) $(ALL_DEPS)
	$(SDACCEL) $(SW_EMUL_TCL)

hw_emulation: $(HW_EMUL_TCL) $(ALL_DEPS)
	$(SDACCEL) $(HW_EMUL_TCL)

board_compile: $(BOARD_COMPILE_TCL) $(ALL_DEPS)
	$(SDACCEL) $(BOARD_COMPILE_TCL)

board_run:
	cd board_compile_solution/pkg/pcie && ./board_compile_solution.exe -d acc -k SgdLR -f test.xclbin

clean:
	rm -f *.jou
	rm -f *.log
	rm -rf .Xil/
	rm -rf sw_solution/
	rm -rf hw_solution/
	rm -rf board_compile_solution/
	rm -rf debug_solution/
