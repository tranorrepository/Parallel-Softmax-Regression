# SDAccel command script
# Design = spam filter (logistic regression + stochastic gradient descent)

# Define a solution name
create_solution -name board_compile_solution -dir . -force

# Source project setup
source scripts/project_setup.tcl

# Compile the design for CPU based emulation
compile_emulation -flow cpu -opencl_binary [get_opencl_binary kernel_file]

# Generate the system estimate report
# report_estimate

# Run the design in CPU emulation mode
run_emulation -flow cpu -args "-d acc -k SgdLR -f kernel_file.xclbin"

# Build application for hardware
build_system

# Package the results for the card
package_system

# Run packaged system
# run_system -args "-d acc -k SgdLR -f kernel_file.xclbin"
