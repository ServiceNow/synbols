from synbols.generate import basic_attribute_sampler
from synbols.motion import dynamic_scene_sampler, update_scene, generate_and_write_dataset
import numpy as np


def transition_function(scene, masks):
    update_scene(scene, masks)
    for symbol in scene.symbols:
        symbol.symbol.translation += 0.05 * np.random.randn(2)
        symbol.speed += 0.01 * np.random.randn(2)
        symbol.angular_speed += 0.01 * np.random.randn(1)
        symbol.symbol.scale += 0.01 * np.random.randn(1)


attr_sampler = basic_attribute_sampler(resolution=(32, 32), n_symbols=1, scale=0.5, max_contrast=False)
scene_sampler = dynamic_scene_sampler(attr_sampler, transition_function=transition_function, time_steps=500)

generate_and_write_dataset('synbols_motion_2', scene_sampler, 1000, preview_shape=(5, 5))
