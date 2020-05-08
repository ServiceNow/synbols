import pandas as pd


def to_pandas(result_str, skip_lines):
    table = []
    for i, line in enumerate(result_str.splitlines()):
        if i in skip_lines:
            continue

        row = []
        for cell in line.split('\t'):
            try:
                if '±' in cell:
                    mean, std = cell.split('±')
                    row.append("%.2f\\%% \pm{%.2f}" % (float(mean) * 100, float(std) * 100))
                else:
                    row.append("%.2f\\%%" % (float(cell) * 100))

            except ValueError:
                row.append(cell)

        table.append(row)
    return pd.DataFrame(table[1:], columns=table[0])


result_str = """\
dataset		mnist	SVHN	default dataset			camouflage
class		digit	digit	char	char	font	char
size		60k	100k	100k	1M	100k	100k
version		-	-				
	N params	accuracy	accuracy	accuracy	accuracy	accuracy	accuracy
MLP		0.9866 ± 0.012	0.8543715427	0.422825	0.7240475	0.0019	0.07115
conv 4		0.9933	0.9149508297	0.879675	0.94809	0.1511	0.797375
resnet 18		0.9951	0.9368085433	0.920425	0.9713625	0.1043	0.837025
resnet 50		0.9954	0.9367701291	0.927925	0.97292	0.0935	0.789125
VGG		0.9956	0.9554394591	0.89855	0.9843525	0.05905	0.941175
WARN (Attention)		0.9966	0.9630070682	0.9686	0.9881425	CUDA MEMORY	0.948675
"""

latex = to_pandas(result_str, skip_lines=[1, 2, 3, 4]).to_latex(index=False, escape=False)

print(latex)
