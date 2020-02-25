#!/usr/bin/env python
# -*- coding: utf-8 -*-

from collections import defaultdict

METADATA = "/usr/share/fonts/truetype/google-fonts/google_fonts_metadata"

# Number of fonts per alphabet
# ----------------------------
# latin : 1011 (done)
# latin-ext : 701 (not sure we need extended yet)
# telugu : 22 (TODO)
# thai : 26 (TODO)
# vietnamese : 210 (mostly latin with accents not sure if there is a big value)
# devanagari : 49 (TODO)
# korean : 24 (done)
# arabic : 20 (TODO)
# cyrillic : 115 (done)
# cyrillic-ext : 89 (not sure we need extended yet)
# greek : 48 (done)
# greek-ext : 36 (not sure we need extended yet)
# hebrew : 17  (TODO)
# khmer : 24 (TODO)
# tamil : 14 (TODO)
# chinese-simplified : 7 (TODO doesn't seem to work with current chinese alphabet)
# gujarati : 9 (maybe)
# bengali : 7(maybe)
# malayalam : 6 (maybe)


# Less than 5
# ------------
# sinhala : 5
# tibetan : 2
# myanmar : 2
# oriya : 2
# lao : 2
# gurmukhi : 4
# ethiopic : 1
# japanese : 1
# kannada : 3

SYMBOL_MAP = {
    'latin': list(u"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"),
    'greek': list(u"ΑαΒβΓγΔδΕεΖζΗηΘθΙιΚκΛλΜμΝνΞξΟοΠπΡρΣσΤτΥυΦφΧχΨψΩω"),
    'cyrillic': list(u"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"),
    'korean': list(u"하지이기리가사자대적어아시장수되전상소부정나인일그주고도히구비치보제스오무생마신서연로내성학실화중공한국해관우여식문미용원"
                   u"의교방바간거음발모경조위저만개세요반물안르차외심분단통유선속계예과불금달입점동감출행산래진양회명재당려초체말러영건강라설집"
                   u"추작남각니피편매근터업버석들절결약직날손배복호표력품없색트활울새머살종청현운타판월면럽름참형확망야쪽임포민역목순별술급올두"
                   u"평년번질데담너늘천드극후파격증디필혼습노최창본접깨길프법루눈환은잠앞십코밤침난워꾸책열독철쓰군락잡벌움처합씨토향삼변능답육"
                   u"키님특먹갈송잘녀험료태"),
    # 'chinese-simplified': list(u"電買開東車紅馬無鳥熱時語假佛德拜黑冰兔妒壤每步聽實證龍賣龜藝戰繩關鐵圖團轉廣惡豐腦雜壓雞價樂氣廳發勞"
    #                            u"劍歲權燒贊兩譯觀營處聲學體點麥蟲舊會萬盜寶國醫雙觸參"),
}


def parse_metadata(file_path):
    alphabet_map = defaultdict(list)
    font_map = defaultdict(list)

    for line in open(file_path, 'r'):
        elements = line.split(',')
        font_name = elements[0].strip()
        for alphabet in elements[1:]:
            alphabet = alphabet.strip()
            alphabet_map[alphabet].append(font_name)
            font_map[font_name].append(alphabet)

    return alphabet_map, font_map


class Alphabet:
    """Combines fonts and symbols for a given language."""

    def __init__(self, name, fonts, symbols):
        self.name = name
        self.symbols = symbols
        self.fonts = fonts


def build_alphabet_map():
    language_map, font_map = parse_metadata(METADATA)

    alphabet_map = {}

    for alphabet_name, font_list in language_map.items():
        if alphabet_name in SYMBOL_MAP.keys():
            alphabet_map[alphabet_name] = Alphabet(alphabet_name, font_list, SYMBOL_MAP[alphabet_name])

    return alphabet_map


ALPHABET_MAP = build_alphabet_map()
