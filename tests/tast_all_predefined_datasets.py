import unittest
from datetime import datetime
from synbols.generate import make_preview
from synbols.predefined_datasets import DATASET_GENERATOR_MAP
from synbols.data_io import pack_dataset
import time as t


n_row, n_col = 10, 10
n_samples = n_row * n_col


class TestPredefinedDatasets(unittest.TestCase):
    def test_all_predefined_datasets(self):
        t0 = t.time()
        for dataset_name, dataset_function in DATASET_GENERATOR_MAP.items():
            print("Generating %s dataset. Info: %s" % (dataset_name, dataset_function.__doc__))
            file_path = '%s_n=%d_%s' % (dataset_name, n_samples, datetime.now().strftime("%Y-%b-%d"))

            ds_generator = dataset_function(n_samples)
            ds_generator = make_preview(ds_generator, file_path + "_preview.png", n_row=n_row, n_col=n_col)

            x, mask, y = pack_dataset(ds_generator)

            self.assertEqual(x.shape[0], n_samples)

        print("The test took %.2fs." % (t.time() - t0))


if __name__ == '__main__':
    unittest.main()