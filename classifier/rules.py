# -*- coding: utf-8 -*-
"""
Правила классификации товара — дословно по постановке задачи 3.

Порядок принятия решения (приоритет зафиксирован постановкой):
  1. Сначала габариты. Если товар меньше 10x10x10 мм или больше
     450x320x320 мм — категория C («Не подходит для сортировки по габаритам»).
     Габаритная проверка ИМЕЕТ ПРИОРИТЕТ над проверкой формы.
  2. Затем форма. Если есть сечение с r_in/R_out >= 0.8 — категория D
     («Не подходит для сортировки без доупаковки»).
  3. Иначе — категория B («Подходит для сортировки»).

Сравнение габаритов выполняется поэлементно после сортировки
(наибольший размер товара против наибольшего лимита и т.д.) — это
эквивалент «влезает ли товар в проём в какой-либо ориентации».

Нештатная логика: если коэффициент круга попадает в зону неопределённости
[0.8 - BORDER_TOL, 0.8 + BORDER_TOL], решение всё равно принимается по
формальному правилу (порог 0.8 строгий), но объект помечается
manual_review_required=True — сигнал исполнительной части/оператору,
что случай пограничный. Отдельной зоны вне B/C/D не вводится.
"""
from __future__ import annotations

from dataclasses import dataclass

LIMIT_MIN = (10.0, 10.0, 10.0)     # мм, по убыванию (все равны)
LIMIT_MAX = (450.0, 320.0, 320.0)  # мм, по убыванию
CIRCLE_THRESHOLD = 0.8
BORDER_TOL = 0.05                  # зона пограничных случаев вокруг порога


@dataclass
class Decision:
    category: str           # 'B' | 'C' | 'D'
    reason: str             # человекочитаемое обоснование
    manual_review: bool     # пограничный случай — залогировать для оператора


def check_dimensions(dims_desc) -> tuple[bool, str]:
    """True, если габариты в допуске основного сортировщика."""
    d = tuple(sorted(dims_desc, reverse=True))
    for i in range(3):
        if d[i] > LIMIT_MAX[i]:
            return False, (
                f'габарит {d[i]:.0f} мм превышает лимит {LIMIT_MAX[i]:.0f} мм '
                f'(позиция {i + 1} по убыванию)')
    for i in range(3):
        if d[i] < LIMIT_MIN[i]:
            return False, (
                f'габарит {d[i]:.1f} мм меньше минимума {LIMIT_MIN[i]:.0f} мм')
    return True, 'габариты в допуске 10x10x10 .. 450x320x320 мм'


def classify(dims_desc, circle_ratio: float | None) -> Decision:
    """Категория товара по правилам постановки. Порядок: габариты -> форма."""
    ok, why = check_dimensions(dims_desc)
    if not ok:
        return Decision('C', why, manual_review=False)

    r = circle_ratio if circle_ratio is not None else 0.0
    borderline = abs(r - CIRCLE_THRESHOLD) <= BORDER_TOL
    if r >= CIRCLE_THRESHOLD:
        return Decision(
            'D', f'круг в сечении: r_in/R_out = {r:.3f} >= {CIRCLE_THRESHOLD}',
            manual_review=borderline)
    return Decision(
        'B', f'{why}; макс. коэффициент круга {r:.3f} < {CIRCLE_THRESHOLD}',
        manual_review=borderline)
