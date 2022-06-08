import argparse
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from glob import glob

import numpy as np
from common import write_image
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", default="", help="The path to scene.")
    parser.add_argument('--render_out', type=str, default='render/')
    parser.add_argument("--exposure", default=0, help="Set amount of exposure applied to render output.")
    args = parser.parse_args()
    return args


def convert_frame(fname, exposure):
    with open(fname, "rb") as f:
        frame = np.load(f)
    write_image(fname, np.clip(frame * (2**exposure), 0.0, 1.0), quality=100)


if __name__ == '__main__':
    args = parse_args()
    futures = []
    with ThreadPoolExecutor() as tp:
        for f in glob(os.path.join(args.scene, args.render_out, "*")):
            futures.append(tp.submit(convert_frame, f, args.exposure))

        for _ in tqdm(as_completed(futures), total=len(futures), desc="Converting"):
            pass
