# SDAccel command script
# Design = spam filter (logistic regression + stochastic gradient descent)

# Define a solution name
open_solution board_compile_solution_backup/board_compile_solution.spr

# Run packaged system
run_system -args "-d acc -k SgdLR -f kernel_file.xclbin"
