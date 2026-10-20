"""Preview images of the built font, rasterized by FreeType (via Pillow)."""
import sys
from PIL import Image, ImageDraw, ImageFont

def render(font_path, lines, out, size=120, margin=40, fg=(90, 36, 18), bg=(233, 196, 122), label=None):
    f = ImageFont.truetype(font_path, size)
    lh = int(size * 1.18)
    widths = [int(f.getlength(l)) for l in lines]
    top = int(size * 0.25) if label else 0
    im = Image.new('RGB', (max(widths) + 2 * margin, len(lines) * lh + 2 * margin + top), bg)
    d = ImageDraw.Draw(im)
    if label:
        d.text((margin, margin * 0.4), label, font=ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', int(size * 0.16)), fill=(80, 60, 40))
    for i, l in enumerate(lines):
        d.text((margin, margin + top + i * lh - int(size * 0.12)), l, font=f, fill=fg)
    im.save(out)

if __name__ == '__main__':
    render('build/Chacara-Regular.otf', sys.argv[2:], sys.argv[1], size=150)
