from pathlib import Path

import dnois

from viflo import conf


def build_index(dir_path: str | Path = None):
    if dir_path is None:
        dir_path = conf.agf_files_dir
    dir_path = Path(dir_path)

    index = {}
    files = dir_path.glob('*.agf', case_sensitive=False)
    for file in files:
        if file.stem not in index:
            index[file.stem] = set()

        with open(file, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
        for line in lines:
            line = line.strip()
            if not line.startswith('NM'):
                continue
            name = line.split()[1]
            index[file.stem].add(name)

    return index


def load_catalog(name: str, dir_path: str | Path = None):
    if dir_path is None:
        dir_path = conf.agf_files_dir
    dir_path = Path(dir_path)

    file_path = dir_path / f'{name}.agf'
    if not file_path.exists():
        file_path = file_path.with_suffix('.AGF')

    dnois.ext.zmx.load_agf(file_path)


def find_material(name: str) -> str:
    original = name
    if ':' in name:
        catalog, name = name.split(':', 1)
    else:
        catalog = None
    if catalog is not None:
        if name in material_index[catalog]:
            return catalog
    else:
        for k, v in material_index.items():
            if name in v:
                return k
    raise ValueError(f'Material {original} not found in any detected catalogs')


material_index = build_index()  # catalog to material map
