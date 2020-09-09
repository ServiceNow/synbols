import os
import platform
import synbols
import sys
from synbols.fonts import LANGUAGE_MAP

print("Hello world from docker!")
print("Synbols installation is at:", synbols)
print("My operating system is:", platform.platform())
print("The language map is:", LANGUAGE_MAP)
print("Supported languages are:", list(LANGUAGE_MAP.keys()))
print("Default english fonts are:", LANGUAGE_MAP["english"].get_alphabet().fonts, len(LANGUAGE_MAP["english"].get_alphabet().fonts))
print("Command line args are:", sys.argv[1:])
