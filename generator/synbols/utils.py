from icu import LocaleData


class Alphabet:
    """Combines fonts and symbols for a given language."""

    def __init__(self, name, fonts, symbols):
        self.name = name
        self.symbols = symbols
        self.fonts = fonts


SYMBOL_MAP = {
    'latin': list(LocaleData("en_US").getExemplarSet()),
    'telugu': list(LocaleData("te").getExemplarSet()),
    'thai': list(LocaleData("th").getExemplarSet()),
    'vietnamese': list(LocaleData("vi").getExemplarSet()),
    'arabic': list(LocaleData("ar").getExemplarSet()),
    'hebrew': list(LocaleData("iw_IL").getExemplarSet()),
    # 'khmer': list(LocaleData("km").getExemplarSet()),  # XXX: see note above
    'tamil': list(LocaleData("ta").getExemplarSet()),
    'gujarati': list(LocaleData("gu").getExemplarSet()),
    'bengali': list(LocaleData("bn").getExemplarSet()),
    'malayalam': list(LocaleData("ml").getExemplarSet()),
    'greek': list(LocaleData("el_GR").getExemplarSet()),
    'cyrillic': list(u"АаБбВвГгДдЕеЁёЖжЗзИиЙйКкЛлМмНнОоПпРрСсТтУуФфХхЦцЧчШшЩщЪъЫыЬьЭэЮюЯя"),
    'korean': list(LocaleData("ko_KR").getExemplarSet()),
    'chinese-simplified': list(LocaleData("zh-CN").getExemplarSet())
}