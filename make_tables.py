import pandas as pd
import numpy as np


def to_latex(result_str, factor=1., skip_rows=(), skip_cols=(), bold_rows=(), bold_cols=(), float_format="%.1f",
             bold_mask=None, red_mask=None):
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

            if (i in bold_rows) or (j in bold_cols) or (bold_mask is not None and bold_mask[i, j]):
                if cell.strip() is not "":
                    cell = "\\bf %s" % cell

            if red_mask is not None and red_mask[i,j]:
                if cell.strip() is not "":
                    cell = "\\rd %s" % cell

            row.append(cell)

        table.append(row)

    n_col = np.max([len(row) for row in table])
    col_format = 'r' + 'c' * (n_col - 1)
    return pd.DataFrame(table[1:], columns=table[0]).to_latex(index=False, escape=False, column_format=col_format)


def supervised_learning():
    result_str = """\
Dataset		MNIST	SVHN	Synbols Default		Camouflage	Korean	Korean	Less Variations	
Label Set		10 Digits	10 Digits	26 Symbols	26 Symbols	26 Symbols	1000 Symbols	1000 Symbols	888 Fonts	888 Fonts
Dataset Size	N params	60k	100k	100k	1M	100k	100k	1M	100k	1M
version		-	-	May 20	May 20	May 20	May 26	May 26	May 20	May 20
		accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy
MLP	72K	98.51 ±0.02	85.04 ±0.21	34.23 ±0.36	74.41 ±0.47	14.70 ±0.20	0.09 ±0.01	22.90 ±0.32	0.12 ±0.02	10.20 ±0.50
Conv-4 Flat	138K	99.32 ±0.06	90.74 ±0.27	79.13 ±0.34	91.75 ±0.02	72.69 ±0.24	5.86 ±2.97	88.02 ±0.47	0.45 ±0.11	29.41 ±0.43
Conv-4 GAP	112K	99.06 ±0.07	88.32 ±0.21	77.84 ±0.48	93.14 ±0.10	65.04 ±0.03	23.94 ±3.71	90.03 ±0.78	6.29 ±0.69	58.77 ±2.62
ResNet-12	7.9M	99.70 ±0.05	96.38 ±0.03	96.11 ±0.05	98.91 ±0.00	94.70 ±0.00	97.35 ±0.40		45.93 ±0.46	95.00 ±0.20
ResNet-12+		99.73 ±0.05	97.19 ±0.04	97.49 ±0.09	99.30 ±0.01	96.90 ±0.05	98.78 ±0.05		67.20 ±0.63	N/A
WRN-28-4		99.64 ±0.06	96.07 ±0.07	95.07 ±0.16	99.07 ±0.01	92.43 ±0.12	97.85 ±0.11		36.29 ±0.50	96.58 ±0.05
WRN-28-4+		99.74 ±0.03	97.30 ±0.05	97.97 ±0.04	99.37 ±0.01	97.39 ±0.08	99.28 ±0.05		80.35 ±0.08	96.09 ±0.01
VGG	138M	99.61 ±0.06	95.32 ±0.25	94.10 ±0.12	98.34 ±0.02	91.86 ±0.09	57.14 ±49.38	98.76 ±0.02	7.83 ±0.60	81.07 ±0.74"""

    mean, std = extract_tables(result_str)
    bold_mask = bold_best(mean, std)
    print(bold_mask)
    print(to_latex(result_str, skip_rows=[3, 4], skip_cols=[1, 8, 10], bold_rows=[0, 1, 2], bold_cols=[0],
                   float_format="%.2f", bold_mask=bold_mask))


def bold_best(mean, std):
    n_rows = mean.shape[0]
    bold_mask = []
    for col_idx in range(mean.shape[1]):
        if np.isnan(mean[:, col_idx]).all():
            print("skip ", col_idx)
            bold_mask.append([False] * n_rows)
            continue

        best_idx = np.nanargmax(mean[:, col_idx])
        print(best_idx)
        threshold = mean[best_idx, col_idx] - 2 * std[best_idx, col_idx]
        print(best_idx, threshold)
        bold_mask.append([val > threshold for val in mean[:, col_idx]])

    return np.array(bold_mask).T


def ood():
    result_str = """Dataset		less variation	Synbols Default							default 1k	default 1M	Less Variations
version		May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20	May 20
Partition		Stratified	\iid	Stratified	Compositional	Stratified	Stratified	Compositional	Stratified	Stratified	Compositional	Stratified
		Char		Font	Char-Font	Scale	Rotation	Rot-Scale	x-Translation	y-Translation	x-y-Translation	Char
class		888 font	char	char	char	char	char	char	char	char	char	888 font
size	N params	1M	100k	100k	100k	100k	100k	100k	100k	100k	100k	100k
MLP		2.35 ±0.03	34.23 ±0.36	33.33 ±0.15	33.65 ±0.38	35.83 ±0.21	22.98 ±0.60	28.23 ±0.66	21.19 ±0.30	24.74 ±0.45	22.97 ±0.65	0.15 ±0.01
Conv-4 Flat		8.49 ±0.18	79.13 ±0.34	78.69 ±0.13	78.93 ±0.29	81.05 ±0.26	62.03 ±0.30	77.88 ±0.22	58.83 ±0.89	64.16 ±0.39	61.21 ±0.14	0.46 ±0.08
Conv-4 GAP		11.26 ±0.20	77.84 ±0.48	77.54 ±0.17	77.58 ±0.38	71.73 ±0.46	57.59 ±0.37	73.31 ±0.35	73.95 ±0.52	74.00 ±0.30	76.41 ±0.52	3.36 ±0.44
Resnet-12		15.05 ±0.32	96.11 ±0.05	95.71 ±0.26	94.78 ±0.20	93.62 ±0.22	84.47 ±0.41	94.23 ±0.11	95.13 ±0.17	94.51 ±0.07	95.59 ±0.13	10.11 ±0.29
Resnet-12+		14.87 ±0.17	97.49 ±0.09	96.44 ±0.10	95.58 ±0.11	96.23 ±0.08	92.86 ±0.07	96.67 ±0.03	97.38 ±0.07		97.60 ±0.08	11.72 ±0.24
WRN-28-4		17.45 ±0.43	95.07 ±0.16	94.96 ±0.12	94.22 ±0.08	92.69 ±0.13	83.02 ±0.51	93.27 ±0.20	94.31 ±0.03	95.93 ±0.13	94.72 ±0.26	9.62 ±0.17
WRN-28-4+		16.34 ±0.40	97.97 ±0.04	96.87 ±0.08	95.94 ±0.10	96.95 ±0.03	93.64 ±0.42	97.26 ±0.12	97.89 ±0.01	95.04 ±0.07	97.97 ±0.06	12.86 ±0.71"""

    mean, std = extract_tables(result_str)
    bold_mask, red_mask = bold_drop(mean, ref_col_idx=3)
    latex_str = to_latex(result_str, skip_rows=[1, 4, 5], skip_cols=[1, 2, 5, 10, 11], bold_rows=[0, 1, 2, 3, 4, 5],
                   bold_cols=[0], bold_mask=bold_mask, red_mask=red_mask, float_format="%.2f")

    print(latex_str.replace("0.", "."))

def bold_drop(mean, ref_col_idx, small_drop=0.5, big_drop=5):
    n_rows = mean.shape[0]
    bold_mask = []
    red_mask = []
    for col_idx in range(mean.shape[1]):
        if np.isnan(mean[:, col_idx]).all() or col_idx == ref_col_idx:
            print("skip ", col_idx)
            bold_mask.append([False] * n_rows)
            red_mask.append([False] * n_rows)

        else:
            drop = mean[:, ref_col_idx] - mean[:, col_idx]

            bold_mask.append(drop < small_drop)
            red_mask.append(drop > big_drop)

    return np.array(bold_mask).T, np.array(red_mask).T


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
Meta-Test	Characters		 Fonts	
Meta-Train	Characters	Fonts	Fonts	Characters
ProtoNet	98.00 ±0.08	77.69 ±0.50	82.26 ±0.91	46.21 ±0.63
RelationNet	93.19 ±1.59	59.61 ±1.72	52.24 ±21.47	31.16 ±9.86
MAML	93.59 ±2.52	72.99 ±0.98	76.42 ±2.90	43.69 ±1.311"""
    print(to_latex(result_str, bold_rows=[0, 1], bold_cols=[0], float_format="%.2f"))


def unsupervised():
    result_str = """\
	Character Accuracy			Font Accuracy		
	Solid Pattern	Shades	Camouflage	Solid Pattern	Shades	Camouflage
Deep InfoMax	89.316 ± 0.302	12.528 ± 0.802	9.368 ± 4.107	18.615 ± 0.305	0.394 ± 0.074	0.361 ± 0.194
VAE	74.608 ± 0.413	39.977 ± 4.141	7.742 ± 0.787	4.163 ± 0.023	0.463 ± 0.091	0.204 ± 0.006
HVAE (2 level)	80.053 ± 1.035	47.763 ± 2.380	8.050 ± 0.206	3.812 ± 0.184	0.534 ± 0.171	0.201 ± 0.006
VAE ResNet	81.845 ± 0.250	52.631 ± 0.224	6.413 ± 0.241	6.227 ± 0.260	0.663 ± 0.047	0.188 ± 0.023
HVAE (2 level) ResNet	79.917 ± 0.023	70.421 ± 1.064	7.117 ± 0.245	4.373 ± 0.028	1.050 ± 0.059	0.213 ± 0.049"""
    print(to_latex(result_str, bold_rows=[0, 1], bold_cols=[0], float_format="%.2f"))


def extract_tables(result_str):
    table = []
    for i, line in enumerate(result_str.splitlines()):

        row = []
        for j, cell in enumerate(line.split('\t')):

            try:
                if '±' in cell:
                    mean, std = [float(s) for s in cell.split('±')]

                else:
                    mean = float(cell)
                    std = 0
            except ValueError:
                mean, std = np.nan, np.nan

            row.append((mean, std))

        table.append(row)
    numeric_values = np.array(table)
    return numeric_values[:, :, 0], numeric_values[:, :, 1]


def ood_merge():
    result_str = """\
0.11 ±0.01	1.73 ±0.18	31.66 ±0.71	0.14 ±0.03	15.69 ±0.20	23.33 ±0.54	25.00 ±0.37	20.83 ±0.31	24.74 ±0.45	19.82 ±0.14
0.45 ±0.08	8.61 ±0.10	78.53 ±0.40	0.50 ±0.22	47.55 ±1.38	62.30 ±0.26	77.98 ±0.36	58.52 ±0.82	64.16 ±0.39	47.39 ±0.85
5.22 ±0.35	11.26 ±0.16	77.39 ±0.34	3.47 ±0.10	71.71 ±0.06	57.44 ±0.14	63.85 ±0.81	73.84 ±0.26	74.00 ±0.30	76.38 ±0.23
4.45 ±0.19	7.93 ±0.13	89.71 ±0.17	3.72 ±0.23	89.56 ±0.09	76.60 ±0.30	87.77 ±0.24	82.96 ±0.15	83.88 ±0.30	83.06 ±0.08
15.49 ±0.70	16.77 ±nan	95.50 ±0.08	11.26 ±0.21	92.96 ±0.31	82.65 ±0.08	93.73 ±0.36	95.77 ±0.22	95.93 ±0.13	96.05 ±0.07
24.18 ±0.61	18.40 ±nan	93.55 ±0.16	15.16 ±0.29	89.62 ±0.12	79.24 ±0.58	87.96 ±0.17	95.08 ±0.02	95.04 ±0.07	92.05 ±0.12
2.18 ±0.28	0.11 ±0.00	93.97 ±0.35	2.18 ±0.28	90.37 ±0.42	80.38 ±0.15	90.87 ±0.42	92.90 ±0.26	92.45 ±0.27	93.45 ±0.33"""

    ref_str = """\
0.16 ±0.01	6.44 ±0.07	72.63 ±0.14	77.82 ±4.58	77.96 ±0.05	69.64 ±0.15	79.65 ±4.46	68.99 ±0.08	69.00 ±0.22	77.27 ±4.80
0.53 ±0.11	30.61 ±0.70	91.13 ±0.19	93.74 ±2.76	91.46 ±0.01	88.30 ±0.10	94.37 ±2.66	89.43 ±0.31	89.33 ±0.04	93.52 ±2.84
6.54 ±0.68	58.33 ±2.67	91.85 ±0.17	94.21 ±2.43	89.93 ±0.09	88.56 ±0.08	94.15 ±2.73	90.89 ±0.08	90.87 ±0.06	93.99 ±2.36
8.86 ±0.95	85.04 ±0.60	97.34 ±0.04	97.97 ±0.90	97.50 ±0.13	95.94 ±0.14	98.18 ±0.86	96.29 ±0.04	96.33 ±0.01	97.77 ±0.85
44.86 ±7.21	95.88 ±0.15	99.25 ±0.04	99.29 ±0.26	98.91 ±0.03	98.46 ±0.03	99.32 ±0.32	98.69 ±0.05	98.70 ±0.02	99.18 ±0.26
76.82 ±1.32	96.95 ±0.03	97.98 ±0.52	98.03 ±0.73	97.87 ±0.41	97.11 ±0.59	98.03 ±0.76	97.38 ±0.53	97.45 ±0.56	97.88 ±0.58
7.14 ±5.21	80.19 ±2.09	98.73 ±0.01	98.87 ±0.48	98.29 ±0.06	97.66 ±0.10	99.00 ±0.46	98.06 ±0.05	98.05 ±0.03	98.76 ±0.49"""

    result = extract_tables(result_str)

    ref = extract_tables(ref_str)

    diff = result[:, :, 0] - ref[:, :, 0]
    std = np.sqrt(result[:, :, 1] ** 2 + ref[:, :, 1] ** 2)

    for i in range(diff.shape[0]):
        row = "\t".join(["%.2f ± %.2f" % (diff[i, j], std[i, j]) for j in range(diff.shape[1])])
        print(row)

    print()
    mean_diff = np.mean(diff, axis=1)
    print("\n".join(["%.2f" % val for val in mean_diff]))

    ref_mean = np.mean(ref[:, 2:, 0], axis=1)
    ref_std = np.std(ref[:, 2:, 0], axis=1)

    print("\n".join(["%.2f ± %.2f" % (m, s) for m, s in zip(ref_mean, ref_std)]))


# supervised_learning()
ood()
# al()
# few_shot()
# unsupervised()

# ood_merge()
