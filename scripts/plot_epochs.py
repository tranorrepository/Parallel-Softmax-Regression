import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties
import sys


import readcsv
import plot_utils



if len(sys.argv) > 1:
    filename = sys.argv[1]

csv = readcsv.parse_to_dir_list(filename)

# all columns in convergenceRate are:
# test FPR,threads_per_datapoint,test TPR,train FPR,name,train TPR,
# train Error,epochs,step_size,test Error,tolerance,datapoints_per_block

# this will represent the different
cols_to_merge_with_name = ["threads_per_datapoint", "datapoints_per_block"]

col_x_axis = "epochs"
cols_y_axis = ["test Error", "train Error"]


for col_y_axis in cols_y_axis:
    lines = {}
    # print(col_y_axis)
    for idx, _ in enumerate(csv["name"]):
        line_name = plot_utils.get_line_name(idx, csv, cols_to_merge_with_name)

        if line_name not in lines:
            lines[line_name] = {'x':[], 'y':[]}

        lines[line_name]['x'].append(csv[col_x_axis][idx])
        lines[line_name]['y'].append(csv[col_y_axis][idx])
    ax = plt.subplot(1, 1, 1)

    plot_utils.plot(lines, col_x_axis, col_y_axis, ax)

plt.show(block=False)

raw_input("press enter to close")

plt.close("all")
