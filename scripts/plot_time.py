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

# all columns in trainAndTest are:
# test FPR,threads_per_datapoint,runs,train Error,name,train TPR,train FPR,
# test TPR,epochs,step_size,test Error,datapoints_per_block,tolerance,train time


# this will represent the different
# cols_to_merge_with_name = ["threads_per_datapoint", "datapoints_per_block"]
cols_to_merge_with_name = ["datapoints_per_block"]

col_x_axis = "threads_per_datapoint"
cols_y_axis = ["train time"]


for col_y_axis in cols_y_axis:
    lines = {}
    # print(col_y_axis)
    for idx, _ in enumerate(csv["name"]):
        line_name = plot_utils.get_line_name(idx, csv, cols_to_merge_with_name)

        if line_name not in lines:
            lines[line_name] = {'x':[], 'y':[]}

        lines[line_name]['x'].append(csv[col_x_axis][idx])
        lines[line_name]['y'].append(csv[col_y_axis][idx])

    plot_utils.adapt_baselines(lines)

    ax = plt.subplot(1, 1, 1)

    plot_utils.plot(lines, col_x_axis, col_y_axis, ax)

plt.show(block=False)

raw_input("press enter to close")

plt.close("all")
