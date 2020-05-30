import pandas as pd
import numpy as np


def to_latex(result_str, factor=1., skip_rows=(), skip_cols=(), bold_rows=(), bold_cols=(), float_format="%.1f"):
    table = []
    for i, line in enumerate(result_str.splitlines()):
        if i in skip_rows:
            continue

        row = []
        for j, cell in enumerate(line.split('\t')):
            if j in skip_cols:
                continue

            try:
                if '±' in cell:
                    mean, std = cell.split('±')
                    format = "%s \std{%s}" % (float_format, float_format)
                    cell = format % (float(mean) * factor, float(std) * factor)
                else:
                    cell = float_format % (float(cell) * factor)

            except ValueError:
                pass

            if (i in bold_rows) or (j in bold_cols):
                if cell.strip() is not "":
                    cell = "\\bf %s" % cell
            row.append(cell)

        table.append(row)

    n_col = np.max([len(row) for row in table])
    col_format = 'r' + 'c' * (n_col - 1)
    return pd.DataFrame(table[1:], columns=table[0]).to_latex(index=False, escape=False, column_format=col_format)


def supervised_learning():
    result_str = """\
Dataset		MNIST	SVHN	Synbols Default		Camouflage	Korean	Less Variations	
Label		10 digit	10 digit	26 char	26 char	26 char	1000 char	888 font	888 font
Dataset Size		60k	100k	100k	1M	100k	100k	100k	1M
version		-	-	May 20	May 20	May 20		May 20	May 20
	N params	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy
MLP		98.52 ±0.02	85.08 ±0.10	32.37 ±0.45	74.58 ±0.49	15.39 ±0.12		0.16 ±0.01	6.44 ±0.07
Conv-4 CAT		99.37 ±0.03	90.91 ±0.14	79.55 ±0.20	92.42 ±0.11	74.93 ±0.12		0.53 ±0.11	30.61 ±0.70
conv-4 GAP		99.13 ±0.05	88.98 ±0.39	78.54 ±0.30	93.58 ±0.07	68.42 ±0.43		6.54 ±0.68	58.33 ±2.67
ResNet-18		99.54 ±0.04	93.37 ±0.20	89.98 ±0.26	97.31 ±0.07	77.54 ±0.49		8.86 ±0.95	85.04 ±0.60
WRN		99.65 ±0.03	96.22 ±0.08	96.07 ±0.11	98.96 ±0.02	95.45 ±0.14		44.86 ±7.21	95.88 ±0.15
WRN+		N/A	N/A	95.34 ±0.10	98.08 ±0.01	94.46 ±0.21		76.82 ±1.32	96.95 ±0.03
VGG		99.65 ±0.02	95.64 ±0.08	94.37 ±0.17	98.26 ±0.02	92.18 ±0.13		7.14 ±5.21	80.19 ±2.09"""
    print(to_latex(result_str, skip_rows=[3, 4], skip_cols=[1], bold_rows=[0, 1, 2], bold_cols=[0], float_format="%.2f"))


def ood():
    result_str = """dataset		less variations	less variation	default 1k	default 1M	default 1k	default 1k	default 1M	default 1k	default 1k	default 1M
version		May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20
Partition		Stratified	Stratified	Stratified	Compositional	Stratified	Stratified	Compositional	Stratified	Stratified	Compositional
		Char	Char	Font	Char-Font	Scale	Rotation	Rot-Scale	x-Translation	y-Translation	x-y-Translation
class		888 font	888 font	char	char	char	char	char	char	char	char
size	N params	100k	1M	100k	100k	100k	100k	100k	100k	100k	100k
MLP		0.11 ±0.01		32.38 ±0.06	32.07 ±0.33	16.93 ±0.24	20.90 ±0.29	27.31 ±0.42	20.68 ±0.14	23.96 ±0.29	22.88 ±0.29
conv 4		0.45 ±0.08		80.74 ±0.41	79.04 ±0.32	64.12 ±0.69	56.89 ±0.61	77.90 ±0.28	58.53 ±0.63	65.45 ±0.20	61.48 ±0.38
conv 4 + gap		5.22 ±0.35		79.39 ±0.31	77.84 ±0.29	70.12 ±0.34	51.11 ±0.57	73.01 ±0.26	75.07 ±0.34	73.66 ±0.19	76.24 ±0.15
resnet 18		4.45 ±0.19		90.27 ±0.21	89.21 ±0.12	81.47 ±0.39	74.48 ±0.16	88.00 ±0.17	84.01 ±0.49	84.92 ±0.13	82.84 ±0.10
WRN		15.49 ±0.70		95.50 ±0.08	94.97 ±0.09	92.96 ±0.31	82.65 ±0.08	94.35 ±0.13	95.77 ±0.22	95.93 ±0.13	95.80 ±0.08
WRN+		24.18 ±0.61		92.94 ±0.11	92.36 ±0.02	89.62 ±0.12	79.24 ±0.58	89.10 ±0.08	94.70 ±0.08	94.90 ±0.10	91.53 ±0.10
VGG		2.18 ±0.28		93.73 ±0.36	92.91 ±0.09	89.52 ±0.39	77.65 ±0.39	90.97 ±0.18	93.22 ±0.42	92.51 ±0.44	93.38 ±0.19"""

    print(to_latex(result_str, skip_rows=[0, 1, 4, 5], skip_cols=[1, 10, 11], bold_rows=[0, 1, 2, 3, 4, 5], bold_cols=[0]))


def al():
    result_str = """	No Noise	Label Noise	Pixel Noise	10\% Missing	Out of the Box	20\% Occluded
BALD	0.56928±0.02216	1.67711±0.06255	0.55845±0.01303	1.43232±0.06358	1.97570±0.05325	1.49566±0.05142
Entropy	0.56370±0.01149	1.74552±0.05587	0.61510±0.01728	2.09894±0.08454	2.26123±0.07979	1.78517±0.06828
Random	0.72460±0.02567	1.81400±0.03548	0.73087±0.02474	1.53829±0.01570	2.09511±0.09883	1.63728±0.05004
BALD Calibrated	0.54926±0.02641	1.66814±0.04546	0.56601±0.04654	1.38180±0.05148	1.95185±0.07983	1.51217±0.04363
Entropy Calibrated	0.60003±0.01829	1.67244±0.03579	0.60302±0.03352	2.10966±0.13357	2.21767±0.05185	1.74065±0.03663"""
    print(to_latex(result_str, bold_rows=[0], bold_cols=[0], float_format="%.2f"))


def few_shot():
    result_str = """\
	test on char		test on font	
	train on char	train on font	train on font	train on char
ProtoNet	96.0	76.2	59.2	42.8
RelationNet	90.1	44.5	41.5	35.0
MAML	94.4	68.6	0.2	38.0"""
    print(to_latex(result_str, bold_rows=[0,1], bold_cols=[0], float_format="%.2f"))



# supervised_learning()
ood()
# al()
# few_shot()