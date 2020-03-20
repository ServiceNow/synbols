from collections import defaultdict
from subprocess import call, check_output
from itertools import chain
import sys

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode


def get_sys_fonts(lang=None):
    """
    Notes:
    ------
        Some fonts may be available in multiple formats so they are associated to multiple ttf fonts. We gather the
        paths of all these files.

    """
    cmd = ['fc-list']
    if lang is not None:
        cmd += [':lang=%s' % lang]
    lines = check_output(cmd).splitlines()

    font_dict = defaultdict(list)
    for font_string in lines:
        font_path, font_string = str(font_string).split(':')[0:2]
        font_path = font_path[2:].strip()
        font_name = ''.join((font_string.split(',')[0].split("\\")[0]).strip().lower().split(' '))
        font_dict[font_name].append(font_path)

    return font_dict


FONT_PATHS = get_sys_fonts()


# TODO: does not work for characters of length 3
def check_font(font, alphabet):
    try:
        font_ok = False  # False until we find one unicode table
        for font_path in FONT_PATHS[font]:
            ttf = TTFont(font_path, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=0)
            for table in ttf["cmap"].tables:
                if table.isUnicode():
                    font_ok = True
                    for char in alphabet:
                        if ord(char) not in table.cmap:
                            # Found a character that is not supported by the font
                            return False
        return font_ok
    except Exception as e:
        print(e)
        print("Problem checking font", font)
        return False