import readcsv
import sys

if len(sys.argv) > 1:
    filename = sys.argv[1]

    pos = filename.rfind('.')
    newfilename = filename[:pos] + '_parsed' + filename[pos:]

    csv = readcsv.parse_to_dir_list(filename)

    with open(newfilename,"w") as f:
        keys = csv.keys()
        f.write(','.join(keys) + '\n')
        for idx in range(len(csv[keys[0]])):
            row = []
            for k in keys:
                row.append(csv[k][idx])

            f.write(','.join(row) + '\n')
