# merge bib/*.bib into allrefs.bib

import os
from pathlib import Path

def merge_bib(bib_dir, out_file):
    bib_files = [f for f in os.listdir(bib_dir) if f.endswith('.bib')]
    with open(out_file, 'w') as out:
        for bib_file in bib_files:
            with open(os.path.join(bib_dir, bib_file), 'r') as f:
                out.write(f.read())
                out.write('\n')

if __name__ == '__main__':

    fpath = os.path.dirname(os.path.abspath(__file__))
    bib_dir = Path(fpath) / 'bib'
    out_file = Path(fpath) / 'allrefs.bib'

    merge_bib(bib_dir, out_file)
    print(f'Merged bib files into {out_file}')