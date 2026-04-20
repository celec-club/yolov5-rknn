import argparse
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def convert_box(size, box):
    # Convert VisDrone box to YOLO xywh box
    dw = 1. / size[0]
    dh = 1. / size[1]
    return (box[0] + box[2] / 2) * dw, (box[1] + box[3] / 2) * dh, box[2] * dw, box[3] * dh


def visdrone2yolo(dir):
    (dir / 'labels').mkdir(parents=True, exist_ok=True)
    pbar = tqdm(list((dir / 'annotations').glob('*.txt')), desc=f'Converting {dir.name}')
    for f in pbar:
        img_size = Image.open((dir / 'images' / f.name).with_suffix('.jpg')).size
        lines = []
        with open(f, 'r') as file:
            for row in [x.split(',') for x in file.read().strip().splitlines()]:
                if row[4] == '0':  # VisDrone 'ignored regions' class 0
                    continue
                cls = int(row[5]) - 1
                box = convert_box(img_size, tuple(map(int, row[:4])))
                lines.append(f"{cls} {' '.join(f'{x:.6f}' for x in box)}\n")
        label_file = dir / 'labels' / f.name
        with open(label_file, 'w') as fl:
            fl.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/VisDrone.yaml', help='dataset yaml path')
    opt = parser.parse_args()

    with open(opt.data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)

    # Resolve dataset root relative to the yaml file location
    yaml_dir = Path(opt.data).parent
    root = (yaml_dir / data_dict['path']).resolve()

    for key in ('train', 'val', 'test'):
        if key in data_dict:
            split_dir = Path(data_dict[key]).parent  # e.g. 'VisDrone2019-DET-train/images' -> 'VisDrone2019-DET-train'
            visdrone2yolo(root / split_dir)
