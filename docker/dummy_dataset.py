'''
We use this file to bootstrap the font and matplotlib caches. This allows faster execution of the synbols command.

'''
from synbols.generate import dataset_generator, basic_attribute_sampler
from synbols.data_io import pack_dataset

attr_sampler = basic_attribute_sampler()
ds_generator = dataset_generator(attr_sampler, 2)
pack_dataset(ds_generator)