from subprocess import call, check_output
from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
import cairo
from synbols import LANGUAGES


def check_char_availability(font_path):
    ttf = TTFont(font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=0)

    len_list = [len(table.cmap.items()) for table in ttf["cmap"].tables]

    ttf.close()
    return len_list


def get_sys_fonts(lang=None):
    cmd = ['fc-list']
    if lang is not None:
        cmd += [':lang=%s' % lang]
    lines = check_output(cmd).splitlines()

    font_dict = {}
    for font_string in lines:
        font_path, font_string = str(font_string).split(':')[0:2]
        font_path = font_path[2:].strip()
        font_name = (font_string.split(',')[0].split("\\")[0]).strip()

        font_dict[font_name] = font_path

    return font_dict


def inspect_font():
    font_dict = get_sys_fonts()
    for font_name, font_path in font_dict.items():
        print("%s (%s)" % (font_name, font_path))
        try:
            print("    ", check_char_availability(font_path))
        except:
            print('error')

            # font_name = list(font_dict.keys())[2]
            # font_path = font_dict[font_name]
            # check_char_availability(font_path)


def print_sys_fonts():
    for font_name, font_path in get_sys_fonts().items():
        print("%s : %s" % (font_name, font_path))


def show_fonts():
    WIDTH, HEIGHT = 2000, 10000
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, WIDTH, HEIGHT)
    ctxt = cairo.Context(surface)

    y_pos = 0

    for language in LANGUAGES.values():
        font_list = language.fonts
        for font in font_list:
            if font.startswith('.'):
                continue
            print("font name:", font)
            ctxt.select_font_face(font, cairo.FONT_SLANT_NORMAL,
                                  cairo.FONT_WEIGHT_NORMAL)
            y_pos += 25
            ctxt.move_to(3, y_pos)

            ctxt.set_font_size(20)
            ctxt.show_text(font + '   ' + ''.join(language.symbols))

    surface.write_to_png("example.png")  # Output to PNG
    call(['open', 'example.png'])


if __name__ == "__main__":
    inspect_font()
    # show_fonts()
    # check_char_availability()
