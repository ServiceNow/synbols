import unittest

from synbols import drawing
from synbols.generate import basic_attribute_sampler


class TestDrawing(unittest.TestCase):
    def test_drawing(self):
        basic_attribute_sampler()
        surface, ctxt = drawing._make_surface(32, 32)
        symbol = drawing.Symbol(alphabet='english', char='c', font='arial', foreground=drawing.SolidColor((0, 0, 0)),
                                is_slant=False, is_bold=False, rotation=0., scale=0.9, translation=(0, 0))

        drawing.draw_symbol(ctxt, symbol)
        array = drawing._surface_to_array(surface)
        self.assertEqual((32, 32, 3), array.shape)


if __name__ == '__main__':
    unittest.main()
