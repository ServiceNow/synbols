from synbols.data_io import write_h5

from synbols.generate import _select, rand_seed
import numpy as np
import types
from tqdm import tqdm

import ffmpeg


def bounce(pix_lower, pix_upper, pos, speed):
    bounce_upper = np.sum(pix_upper > 2) > 2 or pos > 1.
    bounce_lower = np.sum(pix_lower > 2) > 2 or pos < -1.

    if bounce_upper and not bounce_lower:
        return np.abs(speed) * -1

    if bounce_lower and not bounce_upper:
        return np.abs(speed)

    return speed


class Movable:
    def __init__(self, symbol, speed, angular_speed):
        self.symbol = symbol
        self.speed = speed
        self.angular_speed = angular_speed

        self.draw = symbol.draw
        self.make_mask = symbol.make_mask

    def attribute_dict(self):
        dict = self.symbol.attribute_dict()
        dict['speed'] = tuple(self.speed)
        dict['angular_speed'] = float(self.angular_speed)
        return dict

    def move(self, mask):
        self.symbol.translation += self.speed
        self.symbol.rotation += self.angular_speed

        speed_y = bounce(mask[-1, :], mask[0, :], self.symbol.translation[1], self.speed[1])
        speed_x = bounce(mask[:, 0], mask[:, -1], self.symbol.translation[0], self.speed[0])

        self.speed = speed_x, speed_y


def update_scene(scene, masks):
    """Update position and rotation of each symbol according to speed and angular_speed.

    It will use masks to know if a symbols is bouncing on the edge of a scene to adjust the direction of the speed.

    Args:
        scene: Object of type drawing.Image with symbols converted to objects of type Movable.
        masks: masks of each symbol in the image. Used to compute bouncing on walls.
    """

    for i, symbol in enumerate(scene.symbols):
        symbol.move(masks[:, :, i])


def dynamic_scene_sampler(attribute_sampler, transition_function, speed=None, angular_speed=None, time_steps=10):
    def sampler(seed=None):
        _rng = np.random.RandomState(seed)

        scene = attribute_sampler(seed=seed)
        for i in range(len(scene.symbols)):
            _speed = _select(lambda rng: 0.1 * rng.randn(2), speed, _rng)
            _angular_speed = _select(lambda rng: 0.1 * rng.randn(1), angular_speed, _rng)
            scene.symbols[i] = Movable(scene.symbols[i], _speed, _angular_speed)

        out_list = []
        for t in range(time_steps):
            img = scene.make_image()
            scene_attr = scene.attribute_dict()
            masks = scene.make_mask()

            transition_function(scene, masks)

            out_list.append((img, masks, scene_attr))
        img, masks, scene_attr = zip(*out_list)

        return np.stack(img), np.stack(masks), scene_attr

    return sampler


def video_dataset_generator(scene_sampler,
                            n_samples,
                            mask_aggregator=None,
                            dataset_seed=None):
    """High level function generating the dataset from an attribute sampler."""

    if isinstance(scene_sampler, types.GeneratorType):
        scene_generator = scene_sampler

        def sampler(_seed=None):  # ignores the seed
            return next(scene_generator)

        scene_sampler = sampler

    rng = np.random.RandomState(dataset_seed)

    for i in tqdm(range(n_samples)):
        images, masks, attr_seq = scene_sampler(rand_seed(rng))

        if mask_aggregator is not None:
            masks = np.stack([mask_aggregator(mask) for mask in masks])

        yield images, masks, attr_seq


def write_video(file_name, image_seq, frame_rate=10, vcodec='libx264'):
    image_seq = np.asarray(image_seq)
    n_images, height, width, n_channels = image_seq.shape
    node = ffmpeg.input('pipe:', format='rawvideo', pix_fmt='rgb24', s='{}x{}'.format(width, height))
    node = node.output(file_name, pix_fmt='yuv420p', vcodec=vcodec, r=frame_rate)
    node = node.overwrite_output()
    process = node.run_async(pipe_stdin=True)

    for frame in image_seq:
        process.stdin.write(frame.astype(np.uint8).tobytes())
    process.stdin.close()
    process.wait()


def make_video_grid(vid_list, n_row, n_col):
    i = 0
    vid_grid = []
    for col in range(n_col):
        vid_row = []
        for row in range(n_row):
            vid_row.append(vid_list[i])
            i += 1

        vid_grid.append(np.concatenate(vid_row, axis=1))

    vid_grid = np.concatenate(vid_grid, axis=2)

    return vid_grid


def make_preview(generator, file_name, n_row=30, n_col=40):
    """Augment a generator to save a preview when
    the first n_row * n_col images are generated.
    """
    x_list = []
    for x, mask, y in generator:
        if x_list is not None:
            x_list.append(x)
            if len(x_list) == n_row * n_col:
                vid_grid = make_video_grid(x_list, n_row=n_row, n_col=n_col)

                write_video(file_name=file_name, image_seq=vid_grid)

                x_list = None
                tqdm.write("Preview generated.")

        yield x, mask, y


def generate_and_write_dataset(file_path,
                               scene_sampler,
                               n_samples,
                               preview_shape=(10, 10),
                               seed=None):
    """Call the attribute sampler n_samples time to generate a dataset
    and saves it on disk.

    Args:
        file_path: the destination of the dataset an extension
    .h5py will be automatically added.
        scene_sampler: a callable returning objects of type drawing.Image.
        n_samples: integer specifying the number of samples required.
        preview_shape: pair of integers or None.
    Specifies the size of the image grid to render a preview. The png
            will be saved alongside the dataset.
        seed: integer or None. Specifies the seed the random number generator.
    """
    ds_generator = video_dataset_generator(scene_sampler, n_samples, dataset_seed=seed)

    if preview_shape is not None:
        n_row, n_col = preview_shape
        ds_generator = make_preview(ds_generator,
                                    file_path + "_preview.mp4",
                                    n_row=n_row,
                                    n_col=n_col)

    write_h5(file_path + ".h5py", ds_generator, n_samples, split_function=None)
