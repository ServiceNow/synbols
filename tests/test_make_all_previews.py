from datetime import datetime
from synbols.generate import make_preview
from synbols.predefined_datasets import DATASET_GENERATOR_MAP
from synbols.data_io import write_h5
import time as t

def test_make_all_previews():
    n_samples = 25

    t0 = t.time()
    for dataset_name, dataset_function in DATASET_GENERATOR_MAP.items():
        print("Generating %s dataset. Info: %s" % (dataset_name, dataset_function.__doc__))
        file_path = '%s_n=%d_%s' % (dataset_name, n_samples, datetime.now().strftime("%Y-%b-%d"))

        ds_generator = dataset_function(n_samples)
        ds_generator = make_preview(ds_generator, file_path + "_preview.png", n_row=5, n_col=5)
        write_h5(file_path + ".h5py", ds_generator, n_samples)
    print("The test took %.2fs." % (t.time() - t0))
