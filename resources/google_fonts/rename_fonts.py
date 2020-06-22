"""
Adds a "synbols" prefix to the name of each font to make sure that we don't overwrite user fonts
at installation time. The renamed fonts are stored in a new TTF file in the same directory as the
original one.

Usage: python rename_fonts.py font_file.ttf

"""
import os
import sys

from fontTools import ttLib


ttf_path = sys.argv[1]

# Get font name
tt = ttLib.TTFont(ttf_path)
name_records = tt["name"].names

# Determine the font style from name record #2
font_style = ""
for record in name_records:
    if record.nameID == 2:
        font_style = str(record)
        break
if font_style == "":
    sys.stderr.write("Error: unable to find font style in %s%s" % (ttf_path, os.linesep))
    print(ttf_path, "[Failed]")
    exit()

# Determine the font's new name
font_new_name = ""
for record in name_records:
    if record.nameID == 6:
        font_new_name = "synbols-" + str(record).lower()
if font_new_name == "":
    sys.stderr.write("Error: unable to determine font name in %s%s" % (ttf_path, os.linesep))
    print(ttf_path, "[Failed]")
    exit()

# Update the font's name records
for record in name_records:
    if record.nameID == 1:
        # Font family name
        record.string = font_new_name
    elif record.nameID == 4:
        # Full font name
        record.string = font_new_name + " " + font_style
    elif record.nameID == 6:
        # Postscript name
        record.string = font_new_name + "-" + font_style.replace(" ", "")
    elif record.nameID == 16:
        # Preferred Family (Windows only)
        record.string = font_new_name

# Save the new font file
try:
    tt.save(os.path.join(os.path.dirname(os.path.abspath(ttf_path)), font_new_name + ".ttf"))
except Exception as e:
    sys.stderr.write("Error: %s in %s%s" % (str(e), ttf_path, os.linesep))
    print(ttf_path, "[Failed]")
    exit()

print(ttf_path, "[OK]")
