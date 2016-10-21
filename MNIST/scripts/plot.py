import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties
import sys

fontP = FontProperties()
fontP.set_size('small')
marker_types = ['o', 'v', '^', '<', '>', '*']
line_styles = ['-', '--', '-.', ':']

filename = "output/april 25-may 1/test.csv"
if len(sys.argv) > 1:
    filename = sys.argv[1]

with open(filename,"rb") as csv:

    colnames = map(lambda x: x.strip(), csv.readline().split(','))

    num_baselines = 2
    baselines_names = []
    baselines_raw_data = []
    for i in range(0,num_baselines):
        cells = map(lambda x: x.strip(), csv.readline().split(','))
        baselines_names.append(cells[0])
        baselines_raw_data.append([0]+[float(x) for x in cells[1:]])
    baselines_data = np.array(baselines_raw_data)

    names = []
    raw_data = []
    for row in csv:
        if row.strip():
            cells = map(lambda x: x.strip(), row.split(','))
            names.append(cells[0])
            raw_data.append([0]+[float(x) for x in cells[1:]])
    data = np.array(raw_data)


    lines = {}
    num_points_per_line = 4

    # col_of_series = 4 #datapoints per block
    col_of_series = 4 #threads per datapoint
    series_label = colnames[col_of_series]

    # col_of_x_axis = 2 #epochs
    col_of_x_axis = 5 #datapoints per block
    x_label = colnames[col_of_x_axis]

    starting_col_for_figures = 6
    for col_of_y_axis in range(starting_col_for_figures, len(colnames)):
        y_label = colnames[col_of_y_axis]
        for idx, row in enumerate(baselines_data):
            lines[baselines_names[idx]] = {"x":[1, 2**(num_points_per_line-1)], "y":[row[col_of_y_axis], row[col_of_y_axis]]}
        for idx, row in enumerate(data):
            lines[names[idx] + " ( " + str(row[col_of_series]) + " " + colnames[col_of_series] + " )"] = {"x":[], "y":[]}
        for idx, row in enumerate(data):
            lines[names[idx] + " ( " + str(row[col_of_series]) + " " + colnames[col_of_series] + " )"]["x"].append(row[col_of_x_axis])
            lines[names[idx] + " ( " + str(row[col_of_series]) + " " + colnames[col_of_series] + " )"]["y"].append(row[col_of_y_axis])
        plt.figure()
        plt.xscale("log", basex=2)
        plt.title(y_label + " vs. " + x_label)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        handles = []
        i = 0
        for line_name, line_xy in lines.items():
            # hex_color = "#%06X" % random.randint(0,256**3-1)
            plt.plot(line_xy["x"], line_xy["y"], linestyle=line_styles[i % len(line_styles)], marker=marker_types[i % len(marker_types)], alpha=0.7, linewidth=3, label=line_name, markeredgecolor=None)
            i+=1
        plt.legend(prop=fontP, loc='upper center')
        plt.draw()
plt.show(block=False)

raw_input("press enter to close")

plt.close("all")
