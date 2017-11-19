import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.font_manager import FontProperties



fontP = FontProperties()
fontP.set_size('medium')
marker_types = ['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_']
line_styles = ['-', '--', '-.', ':']

baseline_names = ["SGD", "BGD"]


def get_line_name(idx, csv, cols_to_merge_with_name):
    name = csv["name"][idx]

    if name in baseline_names:
        return name
    else:
        suffix = ""
        for colname in cols_to_merge_with_name:
            val = csv[colname][idx]
            suffix += (", " + val + " " + colname.replace('_', ' ').title())
        return csv["name"][idx] + suffix

def plot(lines, col_x_axis, col_y_axis, ax):
    plt.title(col_x_axis.replace('_', ' ').title() + " vs. " + col_y_axis.title())
    plt.xlabel(col_x_axis.replace('_', ' ').title())
    plt.ylabel(col_y_axis.title())
    i = 0
    for line_name, line_xy in lines.items():
        # hex_color = "#%06X" % random.randint(0,256**3-1)
        # print(line_name)
        # print(line_xy["x"])
        # print(line_xy["y"])
        ax.plot(line_xy["x"], line_xy["y"],
            linestyle=line_styles[i % len(line_styles)],
            alpha=0.7, linewidth=3, label=line_name)
        i+=1
    import operator
    handles, labels = ax.get_legend_handles_labels()
    hl = sorted(zip(handles, labels),
                key=operator.itemgetter(1))
    handles2, labels2 = zip(*hl)
    ax.legend(handles2, labels2)
    ax.legend(prop=fontP, loc='upper center')
    plt.draw()

def adapt_baselines(lines):
    min_x = min([min(map(float, lines[k]['x'])) for k in lines])
    max_x = max([max(map(float, lines[k]['x'])) for k in lines])
    print(min_x)
    print(max_x)
    for name in baseline_names:
        lines[name]['x'] = [min_x, max_x]
        lines[name]['y'] = lines[name]['y'] * 2
