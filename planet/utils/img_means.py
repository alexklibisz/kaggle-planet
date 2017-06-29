from sys import argv
from tqdm import tqdm
from PIL import Image
import tifffile as tif
import numpy as np
import os

assert len(argv) == 2, "path to directory required"
assert os.path.exists(argv[1]), "path doesn't exist"

path = argv[1]
files = os.listdir(path)
mean_img = -1

if 'tif' in path:
    get_img = lambda x: tif.imread(x).astype(np.float32)
else:
    get_img = lambda x: np.asarray(Image.open(x).convert('RGB'), np.float32)

for f in tqdm(files):
    img = get_img('%s/%s' % (path, f))
    if type(img) != np.ndarray:
        mean_img = img * 1. / len(files)
    else:
        mean_img += (img * 1. / len(files))

print('image mean = %.16lf' % np.mean(img))

for cidx in range(mean_img.shape[-1]):
    print('channel %d mean = %.16lf' % (cidx, np.mean(img[:, :, cidx])))
