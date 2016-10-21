col_of_names = 0
col_of_config_params = 3

def config_params_to_cols(raw_cell, row_dir, allnames):
    if raw_cell.strip():
        subcells = map(lambda x: x.strip(), raw_cell.split('&'))
        for subcell in subcells:
            param_name, param_val = map(lambda x: x.strip(), subcell.split(':'))
            row_dir[param_name] = param_val
            if param_name not in allnames:
                allnames.append(param_name)

def parse_to_dir_list(filename):

    with open(filename,"rb") as csv:

        colnames = map(lambda x: x.strip(), csv.readline().split(','))

        allnames = [name for name in colnames]

        raw_rows = []
        for str_row in csv:
            if str_row.strip():
                cells = map(lambda x: x.strip(), str_row.split(','))

                row_dir = {}

                for idx, name in enumerate(colnames):
                    if name != 'config params':
                        row_dir[name] = cells[idx]
                    else:
                        config_params_to_cols(cells[idx], row_dir, allnames);

                raw_rows.append(row_dir)

        if 'config params' in allnames:
            allnames.remove('config params')

        new_csv = {}

        for name in allnames:
            new_csv[name] = []
            for row in raw_rows:
                if name in row:
                    new_csv[name].append(row[name])
                else:
                    new_csv[name].append('0')

    return new_csv
