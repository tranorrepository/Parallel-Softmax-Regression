import readcsv
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties
import sys

x_axis_colname = 'epochs'
y_axis_colname = 'test Error'
group_by = 'threads_per_datapoint'

fontP = FontProperties()
fontP.set_size('small')
marker_types = ['o', 'v', '^', '<', '>', '*']
line_styles = ['-', '--', '-.', ':']

def group(csv):
    lines = {}

    keys = csv.keys()
    for idx in range(len(csv[keys[0]])):
        if group_by in csv:
            suffix = csv[group_by][idx]
        else:
            suffix = '--'
        line_name = csv['name'][idx] + '(' + suffix + ' ' + group_by.replace('_', ' ') + ')'

        if line_name not in lines:
            lines[line_name] = {'x':[], 'y':[]}
        else:
            lines[line_name]['x'].append(csv[x_axis_colname][idx])
            lines[line_name]['y'].append(csv[y_axis_colname][idx])

    return lines

def plot_epochs():
    if len(sys.argv) > 1:
        filename = sys.argv[1]

        csv = readcsv.parse_to_dir_list(filename)

        lines = group(csv)

        plt.figure()

        plt.title(x_axis_colname.replace('_', ' ')
            + " vs. "
            + y_axis_colname.replace('_', ' '))

        i = 0
        for line_name, line_xy in lines.items():
            # hex_color = "#%06X" % random.randint(0,256**3-1)
            plt.plot(line_xy["x"], line_xy["y"], linestyle=line_styles[i % len(line_styles)], marker=marker_types[i % len(marker_types)], alpha=0.7, linewidth=3, label=line_name, markeredgecolor=None)
            i+=1

        plt.legend(prop=fontP, loc='upper center')
        plt.draw()

        plt.xlabel(x_axis_colname.replace('_', ' '))
        plt.ylabel(y_axis_colname.replace('_', ' '))

        plt.show(block=False)

        raw_input("press enter to close")

        plt.close("all")

plot_epochs()
