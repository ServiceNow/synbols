from synbols.utils import Language, load_all_languages

print(Language(locale_file="/locales/locale_el_greek.npz").get_alphabet(standard=True, auxiliary=True, lower=True, upper=True, support_bold=True))

print(load_all_languages())