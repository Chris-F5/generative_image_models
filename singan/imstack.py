import PIL.Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-H', '--horizontal', action='store_true')
parser.add_argument('-o', '--output')
parser.add_argument('filenames', nargs='+')
args = parser.parse_args()

images = [PIL.Image.open(fname) for fname in args.filenames]
if args.horizontal:
    width = sum(img.width for img in images)
    height = max(img.height for img in images)
else:
    width = max(img.width for img in images)
    height = sum(img.height for img in images)

stacked_img = PIL.Image.new('RGB', (width, height))

offset = 0
for img in images:
    if args.horizontal:
        stacked_img.paste(img, (offset, 0))
        offset += img.width
    else:
        stacked_img.paste(img, (0, offset))
        offset += img.height

stacked_img.save(args.output)
