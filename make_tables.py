import pandas as pd


def to_pandas(result_str, factor=1., skip_lines=(), skip_cols=(), bold_rows=(), bold_cols=()):
    table = []
    for i, line in enumerate(result_str.splitlines()):
        if i in skip_lines:
            continue

        row = []
        for j, cell in enumerate(line.split('\t')):
            if j in skip_cols:
                continue
            try:
                if '±' in cell:
                    mean, std = cell.split('±')
                    row.append("%.2f \std{%.2f}" % (float(mean) * factor, float(std) * factor))
                else:
                    row.append("%.2f" % (float(cell) * factor))

            except ValueError:
                row.append(cell)

        table.append(row)
    return pd.DataFrame(table[1:], columns=table[0])


result_str = """\
dataset		mnist	SVHN	Synbols Default			camouflage
Label		10 digit	10 digit	52 char	52 char	900 font	52 char
Dataset Size		60k	100k	100k	1M	100k	100k
version		-	-				
	N params	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy
MLP		98.54 ±0.08	85.63 ±0.15	23.79 ±0.33	60.25 ±0.54	0.17 ±0.01	2.10 ±0.02
conv 4		99.41 ±0.03	91.46 ±0.19	71.53 ±0.91	81.13 ±0.04	4.80 ±0.61	27.64 ±0.76
resnet 18		99.54 ±0.06	93.75 ±0.04	74.34 ±0.33	84.75 ±0.07	3.79 ±0.36	43.70 ±0.94
WRN		99.65 ±0.01	96.16 ±0.04	80.86 ±0.43	87.33 ±0.06	nan ±nan	74.83 ±0.23
VGG		99.62 ±0.07	95.74 ±0.21	79.87 ±0.17	85.36 ±0.91	1.33 ±0.18	69.57 ±0.33
WARN		99.64 ±0.06	96.42 ±0.10	80.06 ±0.52	87.17 ±0.17	13.99 ±2.77	72.56 ±1.40
"""

latex = to_pandas(result_str, skip_lines=[3, 4], skip_cols=[1]).to_latex(index=False, escape=False, bold_rows=True)

print(latex)
