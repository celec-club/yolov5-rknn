import argparse
import yaml
import torch
from pathlib import Path

from utils.autoanchor import kmean_anchors
from utils.dataloaders import LoadImagesAndLabels


def run(data='data/VisDrone.yaml', img=1280, n=9, thr=4.0, gen=1000):
    with open(data, errors='ignore') as f:
        data_dict = yaml.safe_load(f)

    root = Path(data_dict['path'])
    train_path = root / data_dict['train']
    dataset = LoadImagesAndLabels(str(train_path), augment=True, rect=True)

    anchors = kmean_anchors(dataset=dataset, n=n, img_size=img, thr=thr, gen=gen, verbose=True)
    print(f'\nNew anchors:\n{anchors.tolist()}')

    # Optionally write anchors back to the yaml
    with open(data, 'r') as f:
        d = yaml.safe_load(f)
    d['anchors'] = anchors.tolist()
    with open(data, 'w') as f:
        yaml.dump(d, f, sort_keys=False)
    print(f'Anchors saved to {data}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/VisDrone.yaml', help='dataset yaml path')
    parser.add_argument('--img', type=int, default=1280, help='image size')
    parser.add_argument('--n', type=int, default=9, help='number of anchors')
    parser.add_argument('--thr', type=float, default=4.0, help='anchor-label wh ratio threshold')
    parser.add_argument('--gen', type=int, default=1000, help='generations for genetic algorithm')
    opt = parser.parse_args()

    run(data=opt.data, img=opt.img, n=opt.n, thr=opt.thr, gen=opt.gen)
