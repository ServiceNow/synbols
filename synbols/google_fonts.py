METADATA = "/usr/share/fonts/truetype/google-fonts/google_fonts_metadata"

LATIN_FONTS = [l.split(', ')[0] for l in open(METADATA, "r") if "latin" in l.split(', ')[1:]]
GREEK_FONTS = [l.split(', ')[0] for l in open(METADATA, "r") if "greek" in l.split(', ')[1:]]