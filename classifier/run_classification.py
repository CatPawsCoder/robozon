# -*- coding: utf-8 -*-
"""
CLI: классификация тестового набора STL по категориям B/C/D.

Запуск (из корня репозитория):
    python -m classifier.run_classification \
        --models data/models --out output

Результаты:
    output/classification.csv   — таблица: габариты, коэффициент круга, категория
    output/sections/<имя>.png   — визуализация решающего сечения каждого объекта
                                  (контур + вписанная и описанная окружности)

Всё детерминировано: одинаковый вход -> одинаковый выход (seed Вельцля
зафиксирован, позиции сечений фиксированы).
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.geometry import analyze
from classifier.rules import classify, CIRCLE_THRESHOLD

AXIS_NAMES = ('X', 'Y', 'Z')


def load_manifest(models_dir: str) -> dict:
    """file -> русское имя (если manifest.csv есть)."""
    path = os.path.join(models_dir, 'manifest.csv')
    names = {}
    if os.path.exists(path):
        with io.open(path, encoding='utf-8') as f:
            for row in csv.DictReader(f):
                names[row['file']] = row['name_ru']
    return names


def plot_section(report, title: str, out_png: str):
    """Решающее сечение: контур, вписанная и описанная окружности."""
    s = report.best
    if s is None:
        return
    fig, ax = plt.subplots(figsize=(6, 6))
    xs, ys = s.hull.exterior.xy
    ax.plot(xs, ys, 'k-', lw=1.5, label='контур сечения (вып. оболочка)')
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(s.center_in[0] + s.r_in * np.cos(th),
            s.center_in[1] + s.r_in * np.sin(th),
            'g-', lw=1.2, label=f'вписанная r={s.r_in:.1f} мм')
    ax.plot(s.center_out[0] + s.r_out * np.cos(th),
            s.center_out[1] + s.r_out * np.sin(th),
            'r--', lw=1.2, label=f'описанная R={s.r_out:.1f} мм')
    verdict = '≥' if s.ratio >= CIRCLE_THRESHOLD else '<'
    ax.set_title(f'{title}\nось {AXIS_NAMES[s.axis]} OBB, позиция '
                 f'{s.position:.2f} — r/R = {s.ratio:.3f} {verdict} '
                 f'{CIRCLE_THRESHOLD}')
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlabel('мм'); ax.set_ylabel('мм')
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description='Классификатор B/C/D по STL')
    ap.add_argument('--models', default='data/models')
    ap.add_argument('--out', default='output')
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    sec_dir = os.path.join(args.out, 'sections')
    os.makedirs(sec_dir, exist_ok=True)

    names = load_manifest(args.models)
    stls = sorted(f for f in os.listdir(args.models)
                  if f.lower().endswith('.stl'))

    rows = []
    for f in stls:
        path = os.path.join(args.models, f)
        report = analyze(path)
        ratio = report.best.ratio if report.best else None
        decision = classify(report.dims_desc, ratio)
        d = report.dims_desc
        title = names.get(f, f)
        rows.append({
            'file': f,
            'name': title,
            'dim1_mm': round(d[0], 1),
            'dim2_mm': round(d[1], 1),
            'dim3_mm': round(d[2], 1),
            'circle_ratio': round(ratio, 3) if ratio is not None else '',
            'section_axis': AXIS_NAMES[report.best.axis] if report.best else '',
            'section_pos': round(report.best.position, 2) if report.best else '',
            'category': decision.category,
            'manual_review_required': decision.manual_review,
            'reason': decision.reason,
        })
        stem = os.path.splitext(f)[0]
        plot_section(report, f'{title} -> {decision.category}',
                     os.path.join(sec_dir, stem + '.png'))
        print(f'{title:<22} {d[0]:6.0f}x{d[1]:3.0f}x{d[2]:3.0f}  '
              f'r/R={ratio if ratio else 0:.3f}  -> {decision.category}'
              f'{"  [BORDERLINE]" if decision.manual_review else ""}')

    csv_path = os.path.join(args.out, 'classification.csv')
    with io.open(csv_path, 'w', encoding='utf-8-sig', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nCSV: {csv_path}')
    print(f'PNG сечений: {sec_dir}')


if __name__ == '__main__':
    main()
