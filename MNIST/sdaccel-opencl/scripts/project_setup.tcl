# Define the target platform of the application
add_device -vbnv xilinx:adm-pcie-7v3:1ddr:2.1

# Host source files
add_files "src/main_sdaccel.cpp"
add_files "src/harness/sdaccel/CLKernel.cpp"
add_files "src/harness/sdaccel/CLMemObj.cpp"
add_files "src/SdaccelSgd.cpp"
add_files "src/utils.cpp"

# Header files
add_files "src/harness/sdaccel/CLKernel.h"
add_files "src/harness/sdaccel/CLMemObj.h"
add_files "src/SdaccelSgd.h"
add_files "src/fileio.hpp"
add_files "src/defs.h"
add_files "src/utils.hpp"

set_property file_type "c header files" [get_files "src/harness/sdaccel/CLKernel.h"]
set_property file_type "c header files" [get_files "src/harness/sdaccel/CLMemObj.h"]
set_property file_type "c header files" [get_files "src/SdaccelSgd.h"]
set_property file_type "c header files" [get_files "src/fileio.hpp"]
set_property file_type "c header files" [get_files "src/defs.h"]
set_property file_type "c header files" [get_files "src/utils.hpp"]

# Kernel definition
create_kernel SgdLR -type clc
# need to change files names here to test other versions of Opencl Kernal
add_files -kernel [get_kernels SgdLR] "src/sgd_serial_kernel.cl"

set_property max_memory_ports true [get_kernels SgdLR]
set_property memory_port_data_width 512 [get_kernels SgdLR]

# Define binary containers
create_opencl_binary kernel_file
set_property region "OCL_REGION_0" [get_opencl_binary kernel_file]
create_compute_unit -opencl_binary [get_opencl_binary kernel_file] -kernel [get_kernels SgdLR] -name ComputeUnit1
