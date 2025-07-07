import argparse
import json
import os

import numpy as np
from matplotlib import pyplot as plt
from tabulate import tabulate


def create_table(src_dir):
    header = [
        "actual distance (mm)",
        "detection rate (%)",
        "measured distance (mm) (% error)",
        "x, y, z (mm)",
        "roll, pitch, yaw (°)",
        "datetime"
    ]
    rows = []
    x = []
    y = []

    roll = []
    pitch = []
    yaw = []

    for root, dirs, files in os.walk(src_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as f:
                    # print(os.path.join(root, file))
                    data = json.load(f)
                    rate = data['stats']['detection_rate'] * 100
                    dist = data['stats']['avg_distance'][0] * 1000
                    pose = ", ".join([f"{p * 1000:.2f}" for p in data['stats']['avg_position'][0]])
                    ori = ", ".join([f"{o:.2f}" for o in data['stats']['avg_abs_orientation'][0]])
                    gt_dist = int(data['args']['note'].split('mm')[0])
                    ref = "_".join(root.split('/')[-1].split('_')[0:2])
                    dist_err = 100 * abs(dist - gt_dist) / gt_dist

                    roll_err = abs(180 - data['stats']['avg_abs_orientation'][0][0])
                    pitch_err = abs(data['stats']['avg_abs_orientation'][0][1])
                    yaw_err = abs(data['stats']['avg_abs_orientation'][0][2])

                    rows.append([gt_dist, f"{rate:.2f}", f"{dist:.2f} ({dist_err:.2f}%)", pose, ori, ref])
                    x.append(gt_dist)
                    y.append(dist_err)
                    roll.append(roll_err)
                    pitch.append(pitch_err)
                    yaw.append(yaw_err)

    rows.sort(key=lambda r:r[0])
    print("Table of results sorted based of the actual distance:")
    print(tabulate(rows, headers=header, tablefmt="github"))

    if args.interpolate:
        print("Interpolated distance error as a function of actual distance:")
        compute_quadratic(x, y, 'dist')


def quadratic_function(x, coefficients):
    a, b, c = coefficients
    return a * x ** 2 + b * x + c


def compute_quadratic(x, y, name):
    paired_list = list(zip(x, y))
    paired_list.sort()
    sorted_x, reordered_y = zip(*paired_list)

    x = list(sorted_x)
    y = list(reordered_y)

    error_coefficients = np.polyfit(x, y, 2)
    print("quadratic coefficients:", error_coefficients)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", default="results", type=str, help="Path to results directory")
    ap.add_argument("--interpolate", action="store_true", help="Interpolate distance error as a function of actual distance")
    args = ap.parse_args()
    create_table(args.input)
