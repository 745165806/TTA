"""Make a dependency-free SVG of the initial objective/score gradient angles."""
import argparse
import csv
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    names = ("U-original", "U-task", "U-mixed")
    colors = ("#326aa8", "#d07025", "#388055")
    values = {name: [] for name in names}
    with args.input.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if (row["subspace"] in values and row["method"] == "ep_tta_guarded"
                    and int(row["steps"]) == 10):
                values[row["subspace"]].append(float(row["gradient_cosine"]))
    if any(len(values[name]) != 512 for name in names):
        raise ValueError("expected exact fixed 512-ID coverage per guarded subspace")
    bins = np.linspace(-1, 1, 21)
    histograms = [np.histogram(values[name], bins=bins)[0] for name in names]
    maximum = max(max(counts) for counts in histograms)
    left, top, width, height = 75, 65, 700, 290
    lines = [
        '<svg xmlns="http://www.w3.org/2000/svg" width="900" height="440" viewBox="0 0 900 440">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="75" y="30" font-size="17">Initial gradient cosine: view loss vs spoof score (R=0)</text>',
        f'<path d="M {left} {top} V {top + height} H {left + width}" fill="none" stroke="#555"/>',
    ]
    for tick in (-1, -.5, 0, .5, 1):
        x = left + width * (tick + 1) / 2
        lines += [f'<path d="M {x:.1f} {top + height} v 5" stroke="#555"/>',
                  f'<text x="{x - 12:.1f}" y="{top + height + 23}" font-size="12">{tick:g}</text>']
    for tick in (0, maximum // 2, maximum):
        y = top + height * (1 - tick / maximum)
        lines += [f'<path d="M {left - 5} {y:.1f} h {width + 5}" stroke="#e4e4e4"/>',
                  f'<text x="38" y="{y + 4:.1f}" font-size="12">{tick}</text>']
    for name, color, counts in zip(names, colors, histograms):
        points = " ".join(f"{left + width * (i + .5) / 20:.1f},{top + height * (1 - count / maximum):.1f}"
                          for i, count in enumerate(counts))
        lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    for i, (name, color) in enumerate(zip(names, colors)):
        x = 115 + 230 * i
        lines += [f'<path d="M {x} 410 h 25" stroke="{color}" stroke-width="3"/>',
                  f'<text x="{x + 33}" y="414" font-size="13">{name}</text>']
    lines.append('</svg>')
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
