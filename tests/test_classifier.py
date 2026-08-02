# -*- coding: utf-8 -*-
"""
Юнит-тесты части определения и классификации товара.

Три уровня:
  1. Правила (rules.py) — чистая логика, без геометрии.
  2. Геометрия (geometry.py) — синтетические примитивы с известными
     аналитическими свойствами (куб, цилиндр, сфера, эллипсоид).
  3. Золотой прогон — регрессия на тестовом наборе организаторов
     (11 STL) против зафиксированных эталонных категорий.

Запуск: pytest tests/ -v
"""
import os
import sys

import numpy as np
import pytest
import trimesh

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from classifier.geometry import (analyze, min_enclosing_circle,
                                 analyze_points, fit_ellipse_ratio)
from classifier.rules import classify, check_dimensions, CIRCLE_THRESHOLD

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(ROOT, 'data', 'models')


# ---------------------------------------------------------------------------
# 1. Правила классификации (без геометрии)
# ---------------------------------------------------------------------------

class TestRules:
    def test_normal_box_is_b(self):
        assert classify((300, 200, 100), 0.5).category == 'B'

    def test_round_within_limits_is_d(self):
        assert classify((300, 90, 90), 0.95).category == 'D'

    def test_oversize_is_c(self):
        assert classify((500, 100, 100), 0.5).category == 'C'

    def test_undersize_is_c(self):
        assert classify((148, 13, 9), 0.5).category == 'C'

    def test_priority_dimensions_over_shape(self):
        # КЛЮЧЕВОЙ тест: негабаритный И круглый -> C (габарит приоритетнее)
        assert classify((489, 489, 264), 0.99).category == 'C'

    def test_threshold_inclusive(self):
        # порог 0.8 включительно -> D
        assert classify((200, 100, 100), CIRCLE_THRESHOLD).category == 'D'
        assert classify((200, 100, 100), CIRCLE_THRESHOLD - 1e-9).category == 'B'

    def test_exact_limit_dimensions_allowed(self):
        ok, _ = check_dimensions((450, 320, 320))
        assert ok
        ok, _ = check_dimensions((450.1, 320, 320))
        assert not ok
        ok, _ = check_dimensions((10, 10, 10))
        assert ok
        ok, _ = check_dimensions((9.9, 10, 10))
        assert not ok

    def test_unsorted_dims_accepted(self):
        # габариты в любом порядке — сортировка внутри
        assert classify((100, 500, 100), 0.5).category == 'C'

    def test_borderline_flag(self):
        assert classify((200, 100, 100), 0.82).manual_review
        assert classify((200, 100, 100), 0.78).manual_review
        assert not classify((200, 100, 100), 0.95).manual_review
        assert not classify((200, 100, 100), 0.5).manual_review


# ---------------------------------------------------------------------------
# 2. Геометрия на синтетических примитивах
# ---------------------------------------------------------------------------

def _analyze_mesh(mesh, tmp_path, name):
    path = str(tmp_path / f'{name}.stl')
    mesh.export(path)
    return analyze(path)


class TestGeometry:
    def test_min_enclosing_circle_square(self):
        pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
        c, r = min_enclosing_circle(pts)
        assert r == pytest.approx(np.sqrt(2) / 2, abs=1e-6)
        assert c == pytest.approx([0.5, 0.5], abs=1e-6)

    def test_min_enclosing_circle_collinear(self):
        pts = np.array([[0, 0], [2, 0], [4, 0]], dtype=float)
        _, r = min_enclosing_circle(pts)
        assert r == pytest.approx(2.0, abs=1e-6)

    def test_box_dims_and_low_ratio(self, tmp_path):
        rep = _analyze_mesh(trimesh.creation.box(extents=[100, 80, 60]),
                            tmp_path, 'box')
        assert rep.dims_desc == pytest.approx((100, 80, 60), abs=0.5)
        # квадратное сечение 80х60: r/R = (30)/(50) = 0.6
        assert rep.best.ratio < 0.75

    def test_rotated_box_obb(self, tmp_path):
        # повернутая коробка: OBB обязан восстановить истинные габариты
        box = trimesh.creation.box(extents=[300, 200, 100])
        box.apply_transform(trimesh.transformations.rotation_matrix(
            0.7, [1, 2, 3]))
        rep = _analyze_mesh(box, tmp_path, 'rbox')
        assert rep.dims_desc == pytest.approx((300, 200, 100), abs=1.0)

    def test_cylinder_is_round(self, tmp_path):
        cyl = trimesh.creation.cylinder(radius=50, height=200, sections=64)
        rep = _analyze_mesh(cyl, tmp_path, 'cyl')
        assert rep.best.ratio > 0.95

    def test_sphere_is_round(self, tmp_path):
        rep = _analyze_mesh(trimesh.creation.icosphere(subdivisions=3,
                                                       radius=100),
                            tmp_path, 'sphere')
        assert rep.best.ratio > 0.95

    def test_ellipse_ratio_analytic(self, tmp_path):
        # эллиптический цилиндр a=50, b=43: лучшее сечение — эллипс,
        # r_in/R_out = b/a = 0.86 (аналитика)
        cyl = trimesh.creation.cylinder(radius=50, height=300, sections=96)
        cyl.apply_scale([1.0, 0.86, 1.0])
        rep = _analyze_mesh(cyl, tmp_path, 'ellcyl')
        assert rep.best.ratio == pytest.approx(0.86, abs=0.03)

    def test_flat_ellipsoid_not_round(self, tmp_path):
        # эллипсоид 100х70х50: лучшее сечение 70/100 = 0.7 < 0.8 -> не круг
        ell = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
        ell.apply_scale([100, 70, 50])
        rep = _analyze_mesh(ell, tmp_path, 'ell')
        assert rep.best.ratio < 0.78

    def test_determinism(self, tmp_path):
        cyl = trimesh.creation.cylinder(radius=40, height=150, sections=48)
        path = str(tmp_path / 'det.stl')
        cyl.export(path)
        r1 = analyze(path).best.ratio
        r2 = analyze(path).best.ratio
        assert r1 == r2


# ---------------------------------------------------------------------------
# 3. Золотой прогон: тестовый набор организаторов
# ---------------------------------------------------------------------------

GOLDEN = {
    'butylka.stl': 'D',
    'cilindr.stl': 'D',
    'korob_300x200x200.stl': 'B',
    'korob_400x400x300.stl': 'C',
    'lunchbox.stl': 'B',
    'meshok.stl': 'D',
    'moyushchee_sredstvo.stl': 'B',
    'pufik.stl': 'C',
    'ruchka.stl': 'C',
    'shlem.stl': 'D',
    'tarelka.stl': 'B/D',  # см. тест
}


@pytest.mark.parametrize('fname,expected', sorted(GOLDEN.items()))
def test_golden_test_set(fname, expected):
    path = os.path.join(MODELS, fname)
    if not os.path.exists(path):
        pytest.skip('тестовый набор не распакован')
    rep = analyze(path)
    decision = classify(rep.dims_desc, rep.best.ratio if rep.best else None)
    if expected == 'B/D':   # тарелка: эталон D
        expected = 'D'
    assert decision.category == expected, (
        f'{fname}: {rep.dims_desc}, r/R='
        f'{rep.best.ratio if rep.best else None} -> {decision.category}, '
        f'ожидалось {expected}')


# ---------------------------------------------------------------------------
# 4. Геометрия восприятия (RGB-D): эллипс-фит по дуге и анализ облака
# ---------------------------------------------------------------------------

class TestPerceptionGeometry:
    def test_ellipse_fit_full(self):
        # полный эллипс a=50, b=43 -> b/a = 0.86, малая невязка
        th = np.linspace(0, 2 * np.pi, 200, endpoint=False)
        pts = np.column_stack([50 * np.cos(th), 43 * np.sin(th)])
        ratio, resid = fit_ellipse_ratio(pts)
        assert ratio == pytest.approx(0.86, abs=0.02)
        assert resid < 0.02

    def test_ellipse_fit_partial_arc(self):
        # верхняя дуга ~230° эллипса (окклюзия низа, как у depth-камеры):
        # b/a восстанавливается из частичной дуги
        th = np.linspace(np.radians(-25), np.radians(205), 120)
        pts = np.column_stack([50 * np.cos(th), 43 * np.sin(th)])
        ratio, resid = fit_ellipse_ratio(pts)
        assert ratio == pytest.approx(0.86, abs=0.05)
        assert resid < 0.02

    def test_ellipse_fit_rejects_line(self):
        # прямой отрезок — не эллипс: либо None, либо большая невязка
        x = np.linspace(-50, 50, 60)
        pts = np.column_stack([x, 0.5 * x + 2])
        ratio, resid = fit_ellipse_ratio(pts)
        assert ratio is None or resid is None or resid > 0.05

    def test_analyze_points_cylinder_cloud(self):
        # облако точек цилиндра -> высокий коэффициент круга
        cyl = trimesh.creation.cylinder(radius=45, height=200, sections=64)
        pts, _ = trimesh.sample.sample_surface(cyl, 6000)
        _, ratio, _ = analyze_points(np.asarray(pts))
        assert ratio > 0.9

    def test_analyze_points_flat_disk_cloud(self):
        # плоский диск (тарелка) -> силуэт-сечение сверху даёт круг
        disk = trimesh.creation.cylinder(radius=100, height=25, sections=64)
        pts, _ = trimesh.sample.sample_surface(disk, 6000)
        _, ratio, _ = analyze_points(np.asarray(pts))
        assert ratio > 0.9

    def test_analyze_points_box_cloud_low(self):
        # коробка -> низкий коэффициент во всех проекциях
        box = trimesh.creation.box(extents=[300, 200, 150])
        pts, _ = trimesh.sample.sample_surface(box, 6000)
        _, ratio, _ = analyze_points(np.asarray(pts))
        assert ratio < 0.8
