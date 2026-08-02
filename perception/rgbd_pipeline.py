# -*- coding: utf-8 -*-
"""
RGB-D контур восприятия: система «видит» товар, а не читает STL.

Измерительная арка из трёх виртуальных depth-камер (верхняя + две боковые
под 45°, как в промышленных DWS-тоннелях) установлена над зоной камеры
(x = 1.0 м) в той же MuJoCo-сцене, что и исполнительная часть.

Контур:
    сцена -> depth-кадры (3 ракурса) -> обратная проекция в облако точек
    -> сегментация от плоскости ленты -> OBB-габариты + метрика
    «круг в сечении» -> правила B/C/D -> (категория для исполнительной части)

Алгоритмическая часть НЕ меняется: используются те же правила
(classifier.rules) и та же метрика сечения (classifier.geometry) —
меняется только источник геометрии (облако точек вместо меша).

Запуск (из корня репозитория):
    python -m perception.rgbd_pipeline

Результаты:
    output/perception/perception_results.csv  — измерения и категории
    output/perception/<объект>_depth.png      — depth-кадр верхней камеры
    output/perception/<объект>_cloud.png      — облако точек (3 проекции)
"""
from __future__ import annotations

import csv
import io
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mujoco
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.geometry import analyze_points
from classifier.rules import classify
from sim.conveyor_sim import (build_mjcf, load_categories, half_heights,
                              BELT_H, X_CAMERA)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'output', 'perception')

CAMS = ('cam_top', 'cam_sideL', 'cam_sideR')
W, H = 960, 720          # разрешение depth-кадра
MAX_DEPTH = 3.0          # м, отсечка фона
# ROI сегментации: зона измерения над лентой
ROI_X = (X_CAMERA - 0.55, X_CAMERA + 0.55)
ROI_Y = (-0.24, 0.24)   # внутри бортов ленты (их грань на +-0.25)
Z_BELT = BELT_H + 0.0065     # плоскость ленты + допуск сегментации, м
MAX_POINTS = 15000           # прореживание объединённого облака


def depth_to_points(model, data, renderer, cam_name: str) -> np.ndarray:
    """Depth-кадр камеры -> точки в мировой СК (метры)."""
    renderer.update_scene(data, camera=cam_name)
    depth = renderer.render()                      # (H, W), метры
    cam_id = model.camera(cam_name).id
    fovy = np.radians(float(model.cam_fovy[cam_id]))
    f = 0.5 * H / np.tan(fovy / 2)                 # фокус в пикселях

    i, j = np.mgrid[0:H, 0:W]
    valid = (depth > 0.01) & (depth < MAX_DEPTH)
    # фильтр «летающих пикселей»: на границе силуэта depth-пиксель
    # смешивает объект и фон и даёт выброс в облаке. Отбрасываем пиксели
    # с большим градиентом глубины — стандартная предобработка depth-камер.
    gy, gx = np.gradient(depth)
    valid &= np.hypot(gx, gy) < 0.015
    z = depth[valid]
    x_cam = (j[valid] + 0.5 - W / 2) * z / f
    y_cam = -(i[valid] + 0.5 - H / 2) * z / f
    pts_cam = np.stack([x_cam, y_cam, -z], axis=1)  # камера смотрит вдоль -Z

    R = data.cam_xmat[cam_id].reshape(3, 3)
    t = data.cam_xpos[cam_id]
    return pts_cam @ R.T + t


def segment_object(points: np.ndarray) -> np.ndarray:
    """Сегментация: ROI над лентой, срез плоскости ленты."""
    m = ((points[:, 0] > ROI_X[0]) & (points[:, 0] < ROI_X[1])
         & (points[:, 1] > ROI_Y[0]) & (points[:, 1] < ROI_Y[1])
         & (points[:, 2] > Z_BELT))
    return points[m]


def capture_cloud(model, data, renderer) -> np.ndarray:
    """Слияние облаков трёх камер + чистка выбросов (SOR).

    Три ракурса (верх + два бока 45°) дают верхнюю дугу профиля товара;
    достройка невидимого низа выполняется отдельно в measure() (отражение
    от плоскости ленты) — здесь только слияние, прореживание и SOR.
    """
    clouds = [segment_object(depth_to_points(model, data, renderer, c))
              for c in CAMS]
    cloud = np.vstack([c for c in clouds if len(c)])
    if len(cloud) > MAX_POINTS:
        idx = np.linspace(0, len(cloud) - 1, MAX_POINTS).astype(int)
        cloud = cloud[idx]
    return statistical_outlier_removal(cloud)


def statistical_outlier_removal(cloud: np.ndarray, k: int = 12,
                                std_ratio: float = 2.0) -> np.ndarray:
    """SOR: отсев точек, чьё среднее расстояние до k соседей заметно
    больше среднего по облаку (стандартная чистка depth-облаков)."""
    if len(cloud) < k + 1:
        return cloud
    from scipy.spatial import cKDTree
    tree = cKDTree(cloud)
    d, _ = tree.query(cloud, k=k + 1)      # +1: сама точка
    mean_d = d[:, 1:].mean(axis=1)
    thr = mean_d.mean() + std_ratio * mean_d.std()
    return cloud[mean_d < thr]


def render_depth_png(model, data, renderer, path: str):
    renderer.update_scene(data, camera='cam_top')
    depth = renderer.render()
    d = np.clip(depth, 0, MAX_DEPTH)
    img = (1 - d / MAX_DEPTH)
    plt.imsave(path, img, cmap='viridis')


def cloud_png(cloud_mm: np.ndarray, title: str, path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for ax, (a, b, la, lb) in zip(axes, [(0, 1, 'X', 'Y'), (0, 2, 'X', 'Z'),
                                         (1, 2, 'Y', 'Z')]):
        ax.scatter(cloud_mm[:, a], cloud_mm[:, b], s=0.4, alpha=0.5)
        ax.set_xlabel(f'{la}, мм'); ax.set_ylabel(f'{lb}, мм')
        ax.set_aspect('equal'); ax.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def measure(model, data, renderer, i, halfz, yaw_deg=0.0):
    """Подать товар i в зону измерения (с разворотом yaw_deg), снять
    depth 3 камерами и вернуть (dims, ratio, borderline, cloud_mm, n)."""
    mujoco.mj_resetData(model, data)
    jadr = model.joint(f'fj{i}').qposadr[0]
    ha = np.radians(yaw_deg) / 2
    data.qpos[jadr:jadr + 7] = [X_CAMERA, 0, BELT_H + halfz[i] + 0.02,
                                np.cos(ha), 0, 0, np.sin(ha)]
    mujoco.mj_forward(model, data)
    while data.time < 1.2:
        mujoco.mj_step(model, data)
    cloud_m = capture_cloud(model, data, renderer)
    if len(cloud_m) < 50:
        return None, None, None, None, len(cloud_m)
    cloud_mm = (cloud_m - [X_CAMERA, 0, BELT_H]) * 1000.0
    # опора на плоскость ленты: низ (невидимый) достраивается отражением
    # видимой верхней части относительно z_max/2 — полный профиль сечения
    # для метрики круга; габариты корректны (низ = плоскость ленты).
    z_top = float(cloud_mm[:, 2].max())
    mirrored = cloud_mm * np.array([1.0, 1.0, -1.0]) + np.array([0, 0, z_top])
    dims, ratio, _ = analyze_points(np.vstack([cloud_mm, mirrored]))
    borderline = abs(ratio - 0.8) <= 0.06
    return dims, ratio, borderline, cloud_mm, len(cloud_m)


def main():
    os.makedirs(OUT, exist_ok=True)
    objects = load_categories()
    halfz = half_heights(objects)
    model = mujoco.MjModel.from_xml_string(build_mjcf(objects))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.enable_depth_rendering()

    rows, ok = [], 0
    for i, row in enumerate(objects):
        dims, ratio, borderline, cloud_mm, n = measure(
            model, data, renderer, i, halfz)
        if dims is None:
            print(f'{row["name"]:<22} НЕ ВИДЕН ({n} точек)')
            continue
        decision = classify(dims, ratio)
        expected = row['category']
        match = decision.category == expected
        ok += match

        stem = os.path.splitext(row['file'])[0]
        render_depth_png(model, data, renderer,
                         os.path.join(OUT, f'{stem}_depth.png'))
        cloud_png(cloud_mm,
                  f'{row["name"]}: {dims[0]:.0f}x{dims[1]:.0f}x{dims[2]:.0f} мм, '
                  f'r/R={ratio:.2f} -> {decision.category}',
                  os.path.join(OUT, f'{stem}_cloud.png'))

        rows.append({
            'name': row['name'],
            'dim1_mm': round(dims[0], 1), 'dim2_mm': round(dims[1], 1),
            'dim3_mm': round(dims[2], 1),
            'circle_ratio': round(ratio, 3),
            'category_rgbd': decision.category,
            'category_reference': expected,
            'match': match,
            'rgbd_borderline': borderline,
            'points': n,
        })
        flag = '  [BORDERLINE->manual_review]' if borderline else ''
        print(f'{row["name"]:<22} {dims[0]:6.0f}x{dims[1]:4.0f}x{dims[2]:4.0f} '
              f'r/R={ratio:.3f} -> {decision.category} '
              f'(эталон {expected}) {"OK" if match else "FAIL"}{flag}')

    csv_path = os.path.join(OUT, 'perception_results.csv')
    with io.open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    reliable = sum(1 for r in rows if r['match'] or r['rgbd_borderline'])
    print(f'\nRGB-D классификация: {ok}/{len(rows)} прямых совпадений '
          f'с эталоном; {reliable}/{len(rows)} с учётом отвода пограничных '
          f'в manual_review')
    print(f'CSV: {csv_path}')


def poses():
    """Устойчивость восприятия к ориентации: каждый товар измеряется при
    нескольких разворотах yaw; проверяется постоянство категории."""
    objects = load_categories()
    halfz = half_heights(objects)
    model = mujoco.MjModel.from_xml_string(build_mjcf(objects))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=H, width=W)
    renderer.enable_depth_rendering()

    angles = [0, 20, 45, 70, 90]
    rows, stable = [], 0
    for i, row in enumerate(objects):
        cats = []
        for ang in angles:
            dims, ratio, _, _, n = measure(model, data, renderer, i, halfz, ang)
            cats.append(classify(dims, ratio).category if dims else '?')
        exp = row['category']
        agree = sum(c == exp for c in cats)
        ok = agree == len(angles)
        stable += ok
        rows.append({'name': row['name'], 'reference': exp,
                     **{f'yaw{a}': c for a, c in zip(angles, cats)},
                     'agree': f'{agree}/{len(angles)}'})
        print(f'{row["name"]:<22} эталон {exp} | '
              f'{" ".join(cats)} | {agree}/{len(angles)} '
              f'{"OK" if ok else "FAIL"}')

    csv_path = os.path.join(OUT, 'perception_poses.csv')
    os.makedirs(OUT, exist_ok=True)
    with io.open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    total = len(objects) * len(angles)
    correct = sum(sum(1 for a in angles
                      if r[f'yaw{a}'] == r['reference']) for r in rows)
    print(f'\nУстойчивость к ориентации: {stable}/{len(objects)} товаров '
          f'стабильны на всех углах; {correct}/{total} измерений корректны')
    print(f'CSV: {csv_path}')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--poses', action='store_true',
                    help='проверка устойчивости к ориентации товара')
    args = ap.parse_args()
    poses() if args.poses else main()
