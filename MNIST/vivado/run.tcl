#=============================================================================
# run.tcl
#=============================================================================
# @brief: A Tcl script for synthesizing the spam filter design.
#
# @desc: This script launches and tests SGD
#

# Project name
set hls_prj spamfilter.prj
# set CFLAGS "-I ../"
set CFLAGS ""

# Open/reset the project
open_project ${hls_prj} -reset
# Top function of the design is "SgdLR"
set_top SgdLR

# Add design and testbench files
# add_files sgd_baseline.cpp -cflags ${CFLAGS}
add_files sgd_serial_kernel.cpp -cflags ${CFLAGS}
add_files -tb utils.cpp -cflags ${CFLAGS}
add_files -tb main_testbench.cpp -cflags ${CFLAGS}

open_solution "solution1"
# Use Zynq device
set_part {xc7z020clg484-1}

# Target clock period is 10ns
create_clock -period 10

# Do not inline update_knn and knn_vote functions
# set_directive_inline -off update_knn
# set_directive_inline -off knn_vote
### You can insert your own directives here ###

############################################

# Simulate the C++ design
csim_design
# Synthesize the design
csynth_design
# Co-simulate the design
cosim_design

#---------------------------------------------
# Collect & dump out results from HLS reports
#---------------------------------------------
set filename "lr_result.csv"
set argv [list $filename $hls_prj]
set argc 2
source "./collect_result.tcl"
