import pymunk
from pymunk import autogeometry as autog
from synbols.motion import video_dataset_generator, dynamic_scene_sampler, update_scene, make_preview
from synbols.generate import basic_attribute_sampler
from synbols import drawing
import numpy as np


def shape_from_mask(body, mask, density=3, elasticity=0.95, friction=0.9, scale=2.):
    sx, sy = mask.shape

    def sample_func(point):
        return float(mask[int(point.x), int(point.y)] > 10)

    poly_set = autog.PolylineSet()
    autog.march_hard(pymunk.BB(0, 0, sx - 1, sy - 1), 15, 15, .5, poly_set.collect_segment,
                     sample_func)

    transform = pymunk.Transform(a=scale / sx, d=scale / sy, tx=-1, ty=-1)
    shapes = []
    for poly in poly_set:
        curve = autog.simplify_curves(poly, 0.5)
        shapes.append(pymunk.Poly(body, curve, transform=transform))

    for shape in shapes:
        shape.density = density
        shape.elasticity = elasticity
        shape.friction = friction

    return shapes


class BoxScene:
    def __init__(self):
        self._space = pymunk.Space()
        self._space.gravity = (0.0, -90.0)
        self._draw_segments()
        self.bodys = None
        self._dt = 1. / 60.
        self._n_physic_steps_per_frame = 1

    def _draw_segments(self):
        points = [(-1.5, -1.5), (-1.5, 1.5), (1.5, -1.5), (1.5, 1.5)]
        segments = []
        for i in range(len(points)):
            seg = pymunk.Segment(self._space.static_body, points[i], points[(i + 1) % 4], 1.0)
            seg.elasticity = 0.95
            seg.friction = 0.9
            segments.append(seg)
            # self._space.add(segments)

    def add_symbols(self, symbols):
        self.bodys = []
        for symbol in symbols:
            body = pymunk.Body()
            mask = symbol.symbol.make_mask((32, 32))
            mask = mask[:, :, 0]
            shapes = shape_from_mask(body, mask)
            for shape in shapes:
                print(shape)

            self._space.add(body, *shapes)
            self.bodys.append(body)
            print('p0:', body.position)

    def update_scene(self, scene, masks):
        if self.bodys is None:
            self.add_symbols(scene.symbols)

        for x in range(self._n_physic_steps_per_frame):
            self._space.step(self._dt)

        for body, symbol in zip(self.bodys, scene.symbols):
            symbol.symbol.translation = np.array(body.position)
            print(body.position)
        print()


box_scene = BoxScene()

attr_sampler = basic_attribute_sampler(resolution=(128, 128), n_symbols=3, scale=0.2)
scene_sampler = dynamic_scene_sampler(attr_sampler, transition_function=box_scene.update_scene, time_steps=100)
generator = make_preview(video_dataset_generator(scene_sampler, 1), "preview.mp4", n_row=1, n_col=1)

x, mask, y = zip(*generator)
