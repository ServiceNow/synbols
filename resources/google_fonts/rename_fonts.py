"""
Adds a "synbols" prefix to the name of each font to make sure that we don't overwrite user fonts
at installation time. The renamed fonts are stored in a new TTF file in the same directory as the
original one. Some metadata about the font and license are also extracted and saved to the same 
directory. This script expects a Google Fonts metadata file in protobuf format.

Usage: python rename_fonts.py font_metadata.pb

Note: This script edits TTF files by modifying specific fields in the name table. Refer to the TTF file specification
      for more details: https://developer.apple.com/fonts/TrueType-Reference-Manual/RM06/Chap6name.html.

"""
import fonts_public_pb2
import os
import sys

from fontTools import ttLib
from google.protobuf import text_format


def rename_font(ttf_path):
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
        raise Exception("Error: unable to find font style in %s%s" % (ttf_path, os.linesep))

    # Determine the font's new name
    font_new_name = ""
    for record in name_records:
        if record.nameID == 1:
                font_new_name = "synbols-" + str(record).lower().replace(" ", "").replace("-", "")
    if font_new_name == "":
        raise Exception("Error: unable to determine font name in %s%s" % (ttf_path, os.linesep))

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
            record.string = (font_new_name + "-" + font_style).replace(" ", "")
        elif record.nameID == 16:
            # Preferred Family (Windows only)
            record.string = font_new_name

    # Save the new font file
    if font_style.lower() != "regular":
        font_new_filename = font_new_name + "-%s.ttf" % font_style
    else:
        font_new_filename = font_new_name + ".ttf"
    tt.save(os.path.join(os.path.dirname(os.path.abspath(ttf_path)), font_new_filename))

    print(ttf_path, "[OK]")
    return font_new_name


def clean_str(s):
    return s.replace(",", " ").replace("\"", "").replace("\n", "")


if __name__ == "__main__":
    font_metadata_path = sys.argv[1]
    font_dir_path = os.path.dirname(font_metadata_path)

    try:
        # Parse Google fonts metadata in protobuf format
        protobuf_file = open(font_metadata_path, 'r')
        protobuf = protobuf_file.read()
        font_family = fonts_public_pb2.FamilyProto()
        text_format.Merge(protobuf, font_family)
    except Exception as e:
        sys.stderr.write("Error: %s in %s%s" % (str(e), font_dir_path, os.linesep))
        print(font_dir_path, "[Failed]")
        exit()

    with open(os.path.join(font_dir_path, "synbols_metadata"), "w") as f_md:
        with open(os.path.join(font_dir_path, "synbols_licenses"), "w") as f_li:
            for font in font_family.fonts:
                alphabets = font_family.subsets

                # Edit TTF file
                ttf_path = os.path.join(font_dir_path, font.filename)
                try:
                    font_new_name = rename_font(ttf_path)
                except Exception as e:
                    sys.stderr.write("Error: %s in %s%s" % (str(e), ttf_path, os.linesep))
                    print(ttf_path, "[Failed]")
                    exit()

                # Output synbols metadata file
                f_md.write("%s, %s\n" % (font_new_name, ", ".join(alphabets)))
                f_li.write("%s, %s, %s, %s, %s\n" % (font_new_name, font.full_name,
                                                     clean_str(font_family.license),
                                                     clean_str(font_family.designer),
                                                     clean_str(font.copyright)))
