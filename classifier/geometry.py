# -*- coding: utf-8 -*-
"""
Геометрический анализ товара для задачи 3 («Робозон»).

Два измерения по STL/мешу:
  1. Габариты — по минимальному ориентированному bounding box (OBB).
  2. «Круг в сечении» — максимум по всем сечениям отношения
     r_вписанной / R_описанной окружности контура сечения.

Определение соответствует постановке задачи:
  «Формальный критерий отнесения товара к объектам с кругом в любом из сечений
   задается через коэффициент сравнения радиусов вписанной и описанной
   окружности, равный 0,8».

Решения по реализации (зафиксированы для воспроизводимости):
  * Сечения берутся перпендикулярно трём осям OBB, N_SLICES позиций на ось
    (равномерно от 5% до 95% длины оси).
  * Контур сечения = выпуклая оболочка точек сечения. Это осознанный выбор:
    правило «круга в сечении» существует потому, что круглые товары КАТЯТСЯ,
    а катится объект по внешней огибающей (бутылка с узким горлом катится
    по корпусу). Выпуклая оболочка устойчива и к не-watertight STL.
  * r_вписанной — точка максимальной удалённости от границы (pole of
    inaccessibility, shapely polylabel).
  * R_описанной — минимальная охватывающая окружность (алгоритм Вельцля).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np
import trimesh
from shapely.geometry import MultiPoint, Polygon
from shapely.ops import polylabel

N_SLICES = 15          # сечений на каждую ось OBB
MIN_SECTION_AREA = 4.0  # мм², отсечка вырожденных сечений


# ---------------------------------------------------------------------------
# Минимальная охватывающая окружность (Вельцль, ожидаемое O(n))
# ---------------------------------------------------------------------------

def _circle_from2(a, b):
    c = (a + b) / 2.0
    return c, float(np.linalg.norm(a - c))


def _circle_from3(a, b, c):
    ax, ay = a; bx, by = b; cx, cy = c
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        return None
    ux = ((ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay)
          + (cx * cx + cy * cy) * (ay - by)) / d
    uy = ((ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx)
          + (cx * cx + cy * cy) * (bx - ax)) / d
    ctr = np.array([ux, uy])
    return ctr, float(np.linalg.norm(a - ctr))


def _in_circle(p, circle):
    c, r = circle
    return np.linalg.norm(p - c) <= r + 1e-9


def min_enclosing_circle(points: np.ndarray, seed: int = 0):
    """Минимальная охватывающая окружность множества 2D-точек (Вельцль)."""
    pts = [np.asarray(p, dtype=float) for p in points]
    rng = random.Random(seed)          # фиксированный seed => воспроизводимо
    rng.shuffle(pts)
    circle = None
    for i, p in enumerate(pts):
        if circle is not None and _in_circle(p, circle):
            continue
        circle = (p, 0.0)
        for j in range(i):
            q = pts[j]
            if _in_circle(q, circle):
                continue
            circle = _circle_from2(p, q)
            for k in range(j):
                s = pts[k]
                if _in_circle(s, circle):
                    continue
                c3 = _circle_from3(p, q, s)
                if c3 is not None:
                    circle = c3
    return circle  # (center, radius)


# ---------------------------------------------------------------------------
# Результаты анализа
# ---------------------------------------------------------------------------

@dataclass
class SectionResult:
    axis: int            # ось OBB (0,1,2), перпендикулярно которой взято сечение
    position: float      # относительная позиция вдоль оси (0..1)
    ratio: float         # r_in / R_out
    hull: Polygon        # контур сечения (выпуклая оболочка)
    r_in: float
    center_in: tuple     # центр вписанной окружности
    r_out: float
    center_out: tuple    # центр описанной окружности


@dataclass
class GeometryReport:
    dims_desc: tuple                 # габариты OBB, мм, по убыванию
    best: SectionResult | None       # сечение с максимальным ratio
    sections: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Основной анализ
# ---------------------------------------------------------------------------

def load_mesh(path: str) -> trimesh.Trimesh:
    m = trimesh.load(path, force='mesh')
    if isinstance(m, trimesh.Scene):
        m = m.dump(concatenate=True)
    return m


def obb_align(mesh: trimesh.Trimesh):
    """Выравнивает меш по осям минимального OBB. Возвращает (меш, габариты)."""
    to_obb, extents = trimesh.bounds.oriented_bounds(mesh)
    m = mesh.copy()
    m.apply_transform(to_obb)
    return m, tuple(float(x) for x in extents)


def section_ratio(mesh: trimesh.Trimesh, axis: int, position: float):
    """Сечение меша плоскостью, перпендикулярной оси OBB; ratio = r_in/R_out."""
    lo = float(mesh.vertices[:, axis].min())
    hi = float(mesh.vertices[:, axis].max())
    coord = lo + (hi - lo) * position
    normal = [0.0, 0.0, 0.0]; normal[axis] = 1.0
    origin = [0.0, 0.0, 0.0]; origin[axis] = coord
    sec = mesh.section(plane_origin=origin, plane_normal=normal)
    if sec is None or len(sec.vertices) < 3:
        return None
    other = [a for a in range(3) if a != axis]
    pts2d = np.asarray(sec.vertices)[:, other]
    hull = MultiPoint([tuple(p) for p in pts2d]).convex_hull
    if hull.geom_type != 'Polygon' or hull.area < MIN_SECTION_AREA:
        return None
    pole = polylabel(hull, tolerance=0.1)
    r_in = float(hull.exterior.distance(pole))
    (c_out, r_out) = min_enclosing_circle(np.asarray(hull.exterior.coords)[:-1])
    if r_out <= 0:
        return None
    return SectionResult(
        axis=axis, position=position, ratio=r_in / r_out, hull=hull,
        r_in=r_in, center_in=(pole.x, pole.y),
        r_out=r_out, center_out=(float(c_out[0]), float(c_out[1])),
    )


def analyze(path: str, n_slices: int = N_SLICES) -> GeometryReport:
    mesh = load_mesh(path)
    aligned, extents = obb_align(mesh)
    dims_desc = tuple(sorted(extents, reverse=True))

    sections: list[SectionResult] = []
    for axis in range(3):
        for t in np.linspace(0.05, 0.95, n_slices):
            res = section_ratio(aligned, axis, float(t))
            if res is not None:
                sections.append(res)

    best = max(sections, key=lambda s: s.ratio) if sections else None
    return GeometryReport(dims_desc=dims_desc, best=best, sections=sections)


# ---------------------------------------------------------------------------
# Анализ облака точек (RGB-D контур): те же габариты и метрика круга,
# но по точкам с depth-камер вместо меша
# ---------------------------------------------------------------------------

def _hull_ratio(points_2d):
    """r_in/R_out выпуклой оболочки множества 2D-точек (мм)."""
    hull = MultiPoint([tuple(p) for p in points_2d]).convex_hull
    if hull.geom_type != 'Polygon' or hull.area < MIN_SECTION_AREA:
        return None, None
    pole = polylabel(hull, tolerance=0.1)
    r_in = float(hull.exterior.distance(pole))
    _, r_out = min_enclosing_circle(np.asarray(hull.exterior.coords)[:-1])
    if r_out <= 0:
        return None, None
    return r_in / r_out, hull


def analyze_points_on_belt(points_mm: np.ndarray, n_slabs: int = 11):
    """Анализ облака точек товара, лежащего на известной плоскости (лента z=0).

    В отличие от analyze_points (3D-OBB на полном облаке), использует
    физическое допущение «товар опирается на ленту»:
      * высота = z_max (нижняя грань в плоскости ленты по определению);
      * след (длина×ширина) = 2D-OBB горизонтальной проекции;
    поэтому габариты корректны даже при полной окклюзии низа товара.
    «Круглость» считается по сечениям в этой опорной СК с аппроксимацией
    эллипса (видимой верхней дуги достаточно).

    points_mm: (N, 3), z отсчитан от плоскости ленты (>= 0).
    Возвращает (dims_desc, best_ratio, best_info).
    """
    p = np.asarray(points_mm, dtype=float)
    height = float(p[:, 2].max())
    to2d, ext2d = trimesh.bounds.oriented_bounds_2D(p[:, :2])
    L, Wd = float(ext2d[0]), float(ext2d[1])
    dims_desc = tuple(sorted([L, Wd, max(height, 1.0)], reverse=True))

    # поворот следа к осям X/Y; вертикаль остаётся Z
    R2 = to2d[:2, :2]
    xy = p[:, :2] @ R2.T
    aligned = np.column_stack([xy, p[:, 2]])
    aligned[:, 0] -= aligned[:, 0].min()
    aligned[:, 1] -= aligned[:, 1].min()

    best_ratio, best_info = 0.0, None
    for axis in range(3):
        lo, hi = aligned[:, axis].min(), aligned[:, axis].max()
        span = hi - lo
        if span < 3.0:
            continue
        slab = max(4.0, min(span / 20.0, 10.0))
        other = [a for a in range(3) if a != axis]
        for t in np.linspace(0.12, 0.88, n_slabs):
            c = lo + span * t
            m = np.abs(aligned[:, axis] - c) < slab / 2
            if m.sum() < 8:
                continue
            sec = aligned[m][:, other]
            ratio, hull = _hull_ratio(sec)
            if ratio is None:
                continue
            e_ratio, resid = fit_ellipse_ratio(sec)
            if (e_ratio is not None and resid is not None
                    and resid < 0.05 and e_ratio > ratio):
                ratio = e_ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_info = (axis, float(t), hull)
    return dims_desc, best_ratio, best_info


def fit_ellipse_ratio(points_2d: np.ndarray):
    """Аппроксимация эллипса по точкам сечения (Халир–Флуссер, устойчивый
    прямой метод) и оценка «круглости».

    Мотивация: depth-камеры видят лишь верхнюю дугу круглого сечения
    (низ закрыт лентой). Дуга эллипса однозначно задаёт полный эллипс,
    поэтому b/a восстанавливается из частичной дуги без достраивания.

    Возвращает (ratio, resid_norm):
      ratio      — b/a аппроксимирующего эллипса (малая/большая полуось),
                   эквивалент r_вписанной/R_описанной для эллипса;
      resid_norm — средняя геометрическая невязка, нормированная на a
                   (мала для настоящих эллиптических дуг, велика для
                    прямых/угловатых контуров — там метрике верить нельзя).
    """
    p = np.asarray(points_2d, dtype=float)
    if len(p) < 6:
        return None, None
    mean = p.mean(axis=0)
    q = p - mean                       # центрируем для обусловленности
    x, y = q[:, 0], q[:, 1]
    D1 = np.column_stack([x * x, x * y, y * y])
    D2 = np.column_stack([x, y, np.ones_like(x)])
    S1 = D1.T @ D1; S2 = D1.T @ D2; S3 = D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None, None
    M = S1 + S2 @ T
    C = np.array([[0, 0, 2.0], [0, -1.0, 0], [2.0, 0, 0]])
    try:
        eigval, eigvec = np.linalg.eig(np.linalg.solve(C, M))
    except np.linalg.LinAlgError:
        return None, None
    cond = 4 * eigvec[0] * eigvec[2] - eigvec[1] ** 2   # условие эллипса
    a1 = eigvec[:, np.nonzero(cond > 0)[0]]
    if a1.size == 0:
        return None, None
    a1 = a1[:, 0].real
    a, b, c = a1
    d, e, f = (T @ a1).real
    # полуоси из коэффициентов A x^2 + B xy + C y^2 + D x + E y + F = 0
    B2 = b * b - 4 * a * c
    if B2 >= 0:
        return None, None
    num = 2 * (a * e * e + c * d * d + f * b * b - b * d * e - 4 * a * c * f)
    s = np.sqrt(max((a - c) ** 2 + b * b, 0.0))
    denom1 = B2 * ((a + c) + s)
    denom2 = B2 * ((a + c) - s)
    if denom1 == 0 or denom2 == 0:
        return None, None
    ax1 = np.sqrt(abs(num / denom1)); ax2 = np.sqrt(abs(num / denom2))
    semi_major, semi_minor = max(ax1, ax2), min(ax1, ax2)
    if semi_major <= 1e-6 or semi_minor < 2.0:
        return None, None
    ratio = semi_minor / semi_major
    # геометрическая невязка: расстояние точки до эллипса ~ (алгебраическое
    # значение) / |градиент|; нормируем на большую полуось
    A, Bc, Cc, Dc, Ec, Fc = a, b, c, d, e, f
    val = (A * x * x + Bc * x * y + Cc * y * y + Dc * x + Ec * y + Fc)
    gx = 2 * A * x + Bc * y + Dc
    gy = Bc * x + 2 * Cc * y + Ec
    g = np.hypot(gx, gy) + 1e-9
    resid = np.abs(val) / g
    return float(ratio), float(np.median(resid) / semi_major)


def analyze_points(points_mm: np.ndarray, n_slabs: int = 11,
                   use_ellipse: bool = True):
    """Габариты OBB и максимальный коэффициент круга по облаку точек.

    points_mm: (N, 3) точки объекта в мм (мировая СК).
    use_ellipse: для каждого сечения брать max(коэф. по оболочке,
      коэф. по аппроксимации эллипса при хорошей невязке) — устойчиво
      к окклюзии нижней части круглого сечения depth-камерами.
    Возвращает (dims_desc, best_ratio, best_info) — совместимо
    с правилами classify().
    """
    pts = np.asarray(points_mm, dtype=float)
    to_obb, extents = trimesh.bounds.oriented_bounds(pts)
    aligned = trimesh.transform_points(pts, to_obb)
    dims_desc = tuple(sorted((float(e) for e in extents), reverse=True))

    def eval_section(sec):
        """Коэффициент круга для набора 2D-точек: max(оболочка, эллипс)."""
        ratio, hull = _hull_ratio(sec)
        if ratio is None:
            return None, None
        if use_ellipse:
            e_ratio, resid = fit_ellipse_ratio(sec)
            if (e_ratio is not None and resid is not None
                    and resid < 0.05 and e_ratio > ratio):
                ratio = e_ratio
        return ratio, hull

    best_ratio, best_info = 0.0, None
    for axis in range(3):
        lo, hi = aligned[:, axis].min(), aligned[:, axis].max()
        span = hi - lo
        if span < 1.0:
            continue
        other = [a for a in range(3) if a != axis]
        # (а) силуэт: полная проекция на плоскость, перпендикулярную оси
        # (плоский диск виден целиком сверху, цилиндр — эллипсом вдоль оси,
        #  коробка — прямоугольником => низкий коэффициент)
        ratio, hull = eval_section(aligned[:, other])
        if ratio is not None and ratio > best_ratio:
            best_ratio, best_info = ratio, (axis, -1.0, hull)
        # (б) тонкие сечения по длине оси
        slab = max(4.0, min(span / 20.0, 10.0))
        for t in np.linspace(0.12, 0.88, n_slabs):
            c = lo + span * t
            m = np.abs(aligned[:, axis] - c) < slab / 2
            if m.sum() < 8:
                continue
            ratio, hull = eval_section(aligned[m][:, other])
            if ratio is not None and ratio > best_ratio:
                best_ratio, best_info = ratio, (axis, float(t), hull)
    return dims_desc, best_ratio, best_info
