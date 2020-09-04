from synbols.utils import Language, load_all_languages

print(Language(locale="en").get_alphabet(standard=True, auxiliary=True, lower=True, upper=True, support_bold=True))

print(load_all_languages())