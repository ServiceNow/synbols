import numpy as np
import os
import pickle

from fontTools.ttLib import TTFont
from fontTools.unicode import Unicode
from itertools import chain
from scipy.sparse import csr_matrix, save_npz
from subprocess import call, check_output
from urllib.request import urlretrieve


UNICODE_STANDARD_URL = 'https://www.unicode.org/Public/UNIDATA/Blocks.txt'


def load_unicode_blocks():
    '''
    Loads all unicode blocks into a dict indexed by name with start/stop indices as value.

    '''
    _path = 'unicode_blocks.txt'
    urlretrieve(UNICODE_STANDARD_URL, _path)
    lines = [l.strip() for l in open(_path, 'r', encoding="utf-8")
             if len(l.strip()) > 0 and '#' not in l]
    lines = {l.split(';')[1].lower(): {'start': int(l.split(';')[0].split('..')[0], base=16),
                                       'stop': int(l.split(';')[0].split('..')[1], base=16)}
             for l in lines}
    os.remove(_path)
    return lines


def get_unicode_tables_by_font(font):
    ttf = TTFont(font, 0, allowVID=0, ignoreDecompileErrors=True, fontNumber=-1)
    supported_chars = set([key for table in ttf["cmap"].tables for key in table.cmap.keys()])
    ttf.close()
    return sorted(supported_chars)


# def get_all_fonts(path):
#     return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.ttf')]

def get_sys_fonts():
    cmd = ['fc-list']
    lines = check_output(cmd).splitlines()

    font_dict = {}
    for font_string in lines:
        font_path, font_string = str(font_string).split(':')[0:2]
        font_path = font_path[2:].strip()
        font_name = (font_string.split(',')[0].split("\\")[0]).strip()

        font_dict[font_name] = font_path

    return font_dict


if __name__ == '__main__':
    unicode_blocks = load_unicode_blocks()

    fonts = get_sys_fonts()

    font_matrix = csr_matrix((max(b['stop'] for b in unicode_blocks.values()), len(fonts)), dtype=np.uint8)

    for i, (name, font) in enumerate(fonts.items()):
        chars = get_unicode_tables_by_font(font)
        font_matrix[chars, i] = 1
        print(name)

    pickle.dump(unicode_blocks, open('unicode_blocks.pkl', 'wb'))
    pickle.dump(fonts, open('fonts.pkl', 'wb'))
    save_npz('font_char_availability.npz', font_matrix)
    
