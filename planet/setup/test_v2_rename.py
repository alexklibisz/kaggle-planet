# Only run this once.
from os import rename
import pandas as pd

df = pd.read_csv('data/test_v2_file_mapping.csv')
test_path = 'data/test-tif-v2'

# Rename all files with a .new extension first.
for old, new in zip(df['old'].values, df['new'].values):
    a = '%s/%s' % (test_path, old)
    b = '%s/%s.new' % (test_path, new)
    print('%-35s -> %s' % (a, b))
    rename(a, b)

# Then rename the .new files without the extension.
for new in df['new'].values:
    a = '%s/%s.new' % (test_path, new)
    b = '%s/%s' % (test_path, new)
    print('%-35s -> %s' % (a, b))
    rename(a, b)
