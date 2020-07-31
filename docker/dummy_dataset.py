from synbols.generate import dataset_generator, basic_image_sampler
from synbols.data_io import pack_dataset

attr_sampler = basic_image_sampler()
ds_generator = dataset_generator(attr_sampler, 2)
pack_dataset(ds_generator)