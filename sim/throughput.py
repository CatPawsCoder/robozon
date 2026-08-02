# -*- coding: utf-8 -*-
"""
Масштабирование ПАК на непрерывный поток (дискретно-событийная модель, SimPy).

Отвечает на требование постановки/экспертов: решение должно работать в
НЕПРЕРЫВНОМ потоке, а не только на разовом сценарии. Здесь расчётом и
симуляцией показано, как узел держит поток и как масштабируется каскадом.

Модель участка:
  * товары поступают по ленте (1 м/с) со средним шагом s -> темп λ = v/s;
  * доля B («подходит») проходит НАСКВОЗЬ в основной сортировщик и НЕ
    занимает лоток; активно обслуживаются только C и D (доля ρ = P(C)+P(D));
  * накопитель = очередь (буфер) перед лотком(ами);
  * N лотков-серверов, время занятости лотка на один товар T_s
    (приём→поворот→опрокидывание→возврат; из циклограммы ≈ 3.6 с;
     полный end-to-end cycle time ~5.3 с включает транспорт до лотка).
  * условие устойчивости (нет переполнения): λ · ρ < N / T_s.

Режимы прибытия: детерминированный (регулярный шаг на ленте, базовый) и
пуассоновский (пачки, стресс). Всё детерминировано при фиксированном seed.

Запуск:
    python -m sim.throughput            # таблица + график output/throughput.png
"""
from __future__ import annotations

import csv
import io
import os
import random

import simpy

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

V_BELT = 1.0          # м/с (постановка)
T_SERVICE = 3.6       # с, занятость лотка на 1 товар (приём→возврат)
REJECT_FRACTION = 0.35   # доля C+D в реальном потоке (B проходит насквозь)
SIM_TIME = 3600.0     # с (1 час модельного времени)
BUFFER = 20           # ёмкость накопителя (шт.)


def run(lam, rho=REJECT_FRACTION, n_trays=1, t_service=T_SERVICE,
        buffer=BUFFER, sim_time=SIM_TIME, arrivals='deterministic', seed=0):
    """Одна конфигурация. Возвращает метрики непрерывной работы."""
    rng = random.Random(seed)
    env = simpy.Environment()
    trays = simpy.Resource(env, capacity=n_trays)
    accumulator = simpy.Container(env, capacity=buffer, init=0)

    stats = {'arrived': 0, 'to_tray': 0, 'passed_B': 0, 'served': 0,
             'lost_overflow': 0, 'wait_sum': 0.0, 'qmax': 0, 'busy_time': 0.0}

    def handle(item_id):
        # занять место в накопителе (переполнение = потеря/затор)
        if accumulator.level >= buffer:
            stats['lost_overflow'] += 1
            return
        yield accumulator.put(1)
        stats['qmax'] = max(stats['qmax'], accumulator.level)
        t0 = env.now
        with trays.request() as req:
            yield req
            stats['wait_sum'] += env.now - t0
            yield accumulator.get(1)
            yield env.timeout(t_service)   # лоток занят
            stats['busy_time'] += t_service
            stats['served'] += 1

    def source():
        i = 0
        while True:
            if arrivals == 'poisson':
                yield env.timeout(rng.expovariate(lam))
            else:
                yield env.timeout(1.0 / lam)      # регулярный шаг
            i += 1
            stats['arrived'] += 1
            if rng.random() < rho:               # C или D -> на лоток
                stats['to_tray'] += 1
                env.process(handle(i))
            else:
                stats['passed_B'] += 1           # B -> насквозь

    env.process(source())
    env.run(until=sim_time)

    util = stats['busy_time'] / (n_trays * sim_time)
    tray_load = stats['to_tray']
    thr_total = stats['arrived'] / sim_time * 60.0        # шт/мин на входе
    lost = stats['lost_overflow']
    stable = lost == 0 and util < 0.98
    mean_wait = stats['wait_sum'] / stats['served'] if stats['served'] else 0.0
    return {
        'lambda_s': round(lam, 3), 'feed_min': round(lam * 60, 1),
        'rho': rho, 'n_trays': n_trays,
        'util': round(util, 3), 'qmax': stats['qmax'],
        'mean_wait_s': round(mean_wait, 2),
        'served': stats['served'], 'to_tray': tray_load,
        'lost_overflow': lost, 'stable': stable,
    }


def main():
    os.makedirs(os.path.join(ROOT, 'output'), exist_ok=True)
    mu1 = 1.0 / T_SERVICE                      # производительность 1 лотка
    print(f'Время занятости лотка T_s = {T_SERVICE} с  ->  один лоток '
          f'обслуживает {mu1*60:.1f} reject-товаров/мин')
    print(f'Доля активного изъятия (C+D): ρ = {REJECT_FRACTION}\n')

    # разворот по темпу подачи и числу лотков
    feeds = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]      # товаров/с на входе (все категории)
    rows = []
    print(f'{"вход,шт/мин":>12} {"λ·ρ,шт/с":>9} {"лотков":>7} '
          f'{"загрузка":>9} {"очередь":>8} {"ожид,с":>7} {"устойчиво":>10}')
    for lam in feeds:
        # минимально необходимое число лотков по аналитике: N > λ·ρ·T_s
        need = lam * REJECT_FRACTION * T_SERVICE
        for n in range(1, 5):
            r = run(lam, n_trays=n)
            rows.append(r)
            mark = 'да' if r['stable'] else 'НЕТ (переполн.)'
            print(f'{r["feed_min"]:>12} {lam*REJECT_FRACTION:>9.2f} {n:>7} '
                  f'{r["util"]*100:>8.1f}% {r["qmax"]:>8} {r["mean_wait_s"]:>7} '
                  f'{mark:>10}')
            if r['stable']:
                break      # нашли минимально достаточный каскад
        print(f'   (аналитика: нужно N > {need:.2f} лотка)')

    csv_path = os.path.join(ROOT, 'output', 'throughput.csv')
    with io.open(csv_path, 'w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    # график: макс. устойчивый темп vs число лотков
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        Ns = [1, 2, 3, 4]
        max_feed = [n / (REJECT_FRACTION * T_SERVICE) * 60 for n in Ns]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        ax.bar([str(n) for n in Ns], max_feed, color='#2D7DD2')
        for n, v in zip(Ns, max_feed):
            ax.text(n - 1, v + 5, f'{v:.0f}', ha='center', fontsize=11, weight='bold')
        ax.set_xlabel('Число лотков (каскад)')
        ax.set_ylabel('Предел подачи (насыщение), шт/мин')
        ax.set_title(f'Масштабирование на непрерывный поток '
                     f'(ρ={REJECT_FRACTION}, T_s={T_SERVICE} с)')
        ax.grid(axis='y', alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(ROOT, 'output', 'throughput.png'), dpi=110)
        print(f'\nГрафик: output/throughput.png')
    except ImportError:
        pass

    print(f'CSV: {csv_path}')
    # стресс: пуассоновский поток (пачки) на устойчивом режиме
    r = run(2.0, n_trays=3, arrivals='poisson')
    print(f'\nСтресс (пуассон, λ=2/с, 3 лотка): очередь_max={r["qmax"]}, '
          f'потерь={r["lost_overflow"]}, загрузка={r["util"]*100:.0f}%')


if __name__ == '__main__':
    main()
