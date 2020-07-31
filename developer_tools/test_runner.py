import os
import platform
import synbols
import sys
from synbols.fonts import ALPHABET_MAP

print("Hello world from docker!")
print("Synbols installation is at:", synbols)
print("My operating system is:", platform.platform())
print("Here are a few installed google fonts:", os.listdir("/usr/share/fonts/truetype/google-fonts")[:10])
print("The alphabet map is:", ALPHABET_MAP)
print("Command line args are:", sys.argv[1:])
