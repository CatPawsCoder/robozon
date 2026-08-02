# -*- coding: utf-8 -*-
"""
Цифровой двойник участка предварительной сортировки (задача 3, «Робозон»).

Схема (вид сверху, размеры участка 6000x10000 мм из схемы площадки):

      A ================== [камера] ====== [накопитель: tilt-tray] --> B (прямо)
        подающий конвейер, лента 1 м/с          |     \
        ширина 500 мм, высота 700 мм            v      v
                                              C (влево)  D (вправо)

Исполнительный узел — НАКОПИТЕЛЬ-ЛОТОК (tilt-tray) на двух приводах:
  * приём: лоток почти горизонтален (контруклон RECEIVE_TILT), товар
    сходит с ленты, гасит скорость о контруклон и останавливается;
  * поворот: сервопривод yaw доворачивает лоток на угол зоны (B/C/D);
  * опрокидывание: сервопривод tilt наклоняет лоток на DUMP_TILT,
    товар соскальзывает в зону обработки.
Товар неподвижен в момент поворота — исключён перелет крупного товара
через борт (боковой момент = 0). Это классическая схема tilt-tray
сортеров в промышленности.

Связка частей ПАК: категории читаются из output/classification.csv —
результата модуля определения и классификации. Контроллер получает
(объект, категория) в момент прохождения товаром позиции камеры и
синхронизирует работу по известной скорости ленты (1 м/с):
    t_прибытия = t_камеры + (x_лотка - x_камеры) / v_ленты
Прибытие товара в лоток контролируется по расчётному времени: если товар
не пришёл к t_прибытия + резерв — фиксируется событие «jam» (затор).

Запуск (из корня репозитория):
    python -m sim.conveyor_sim                # headless, лог в output/sim_log.csv
    python -m sim.conveyor_sim --video        # + запись кадров MP4

Нештатные случаи: объект с manual_review_required маршрутизируется по
формальному правилу с записью события; затор на подаче -> событие jam;
потеря устойчивости контакта -> аварийный съём объекта с линии (estop).
"""
from __future__ import annotations

import argparse
import csv
import io
import os

import mujoco
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- геометрия участка (метры) ---------------------------------------------
BELT_LEN = 3.0        # моделируемый отрезок подающего конвейера
BELT_W = 0.5          # ширина ленты 500 мм (постановка)
BELT_H = 0.7          # высота ленты 700 мм (постановка)
BELT_V = 1.0          # скорость ленты 1 м/с (постановка)
X_SPAWN = 0.25        # точка появления товара на ленте
X_CAMERA = 1.0        # позиция «камеры» (зона обзора CV)
X_CHUTE = BELT_LEN + 0.05   # шарнир лотка у конца ленты
TRAY_Z = BELT_H - 0.17      # ложе лотка ниже среза ленты (приём падением)

RECEIVE_TILT = 0.0    # приём: ложе горизонтально, товар гасится V-формой
DUMP_TILT = 36.0      # угол опрокидывания, град
YAW = {'B': 0.0, 'C': +55.0, 'D': -55.0}   # угол поворота лотка на зону
INFER_TIME = 0.15     # бюджет на инференс CV, с
JAM_RESERVE = 2.0     # резерв ожидания прибытия товара в лоток, с

SETTLE_TIME = 3.0     # ожидание после входа в зону, с
DT = 0.001


def half_heights(objects) -> list[float]:
    """Полувысота hull каждого объекта (для мягкой подачи на ленту)."""
    import trimesh
    out = []
    for row in objects:
        stl = os.path.join(ROOT, 'data', 'models_sim', row['file'])
        b = trimesh.load(stl, force='mesh').bounds
        out.append(float(b[1][2] - b[0][2]) * 0.5e-3)
    return out


def load_categories() -> list[dict]:
    """Читает результат части определения и классификации товара."""
    path = os.path.join(ROOT, 'output', 'classification.csv')
    with io.open(path, encoding='utf-8-sig') as f:
        return list(csv.DictReader(f))


def build_mjcf(objects: list[dict]) -> str:
    """Собирает MJCF-сцену: лента, tilt-tray, зоны-накопители, STL-товары."""
    import trimesh
    assets, bodies = [], []
    for i, row in enumerate(objects):
        # выпуклые оболочки: MuJoCo в любом случае строит hull для коллизий
        stl = os.path.join(ROOT, 'data', 'models_sim', row['file'])
        # реалистичная масса: лёгкая тара ~150 кг/м3 по hull, но не меньше 150 г
        # (сверхлёгкие тонкие объекты дестабилизируют контактный решатель)
        vol = float(trimesh.load(stl, force='mesh').volume) * 1e-9  # м³
        density = max(150.0, 0.15 / max(vol, 1e-6))
        assets.append(
            f'<mesh name="obj{i}" file="{stl}" '
            f'scale="0.001 0.001 0.001"/>')
        # до своей очереди товары «запаркованы» вне участка
        bodies.append(f'''
        <body name="obj{i}" pos="{-2 - i * 0.8:.2f} 3.0 0.3">
          <freejoint name="fj{i}"/>
          <geom type="mesh" mesh="obj{i}" density="{density:.0f}"
                friction="0.25 0.005 0.002" rgba="0.8 0.6 0.2 1"/>
        </body>''')

    def bin_walls(name, cx, cy, hx, hy, open_side):
        """Накопитель зоны (аналог ролл-кейджа): 3 стенки, открыт к лотку."""
        h, t = 0.3, 0.02
        walls = {
            'x-': (cx - hx, cy, t, hy), 'x+': (cx + hx, cy, t, hy),
            'y-': (cx, cy - hy, hx, t), 'y+': (cx, cy + hy, hx, t)}
        del walls[open_side]
        return ''.join(
            f'<geom name="{name}_{k}" type="box" pos="{px:.2f} {py:.2f} {h/2}" '
            f'size="{sx:.2f} {sy:.2f} {h/2}" rgba="0.5 0.5 0.55 1"/>'
            for k, (px, py, sx, sy) in walls.items())

    # --- лоток: V-образное ложе (tray-cradle) в СК наклоняемой рамы ------
    # Две плиты под +/-12 град сходятся в ложбину (local x ~0.5):
    # товар вкатывается с ленты, гасит скорость о встречный склон
    # и оседает в ложбине (в т.ч. круглый — V запирает качение).
    # При опрокидывании (+DUMP_TILT) дальний склон становится спуском,
    # товар сходит через открытый дальний край. Ближний бортик страхует
    # от отката на конвейер, боковые стенки — от бокового схода.
    chute_geoms = (
        # ближний склон (вниз к ложбине; рим отодвинут под мостик)
        '<geom type="box" pos="0.272 0 0.047" euler="0 12 0" '
        'size="0.225 0.35 0.015" rgba="0.2 0.5 0.8 1" density="300" '
        'friction="0.25 0.005 0.0001"/>'
        # дальний склон круче (18), длиннее и ВНАХЛЁСТ под ближним:
        # стык плит закрыт — тонкий товар не проваливается в шов
        '<geom name="tray" type="box" pos="0.764 0 0.087" euler="0 -18 0" '
        'size="0.320 0.35 0.015" rgba="0.2 0.5 0.8 1" density="300" '
        'friction="0.25 0.005 0.0001"/>'
        # боковые стенки (с отступом от шарнира — зазор при повороте)
        '<geom type="box" pos="0.700 0.365 0.115" size="0.400 0.015 0.075" '
        'rgba="0.2 0.5 0.8 1"/>'
        '<geom type="box" pos="0.700 -0.365 0.115" size="0.400 0.015 0.075" '
        'rgba="0.2 0.5 0.8 1"/>')

    return f'''
<mujoco model="robozone_task3">
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" impratio="10"/>
  <default>
    <geom solref="0.005 1"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
  </visual>
  <asset>
    {''.join(assets)}
    <texture name="grid" type="2d" builtin="checker" width="256" height="256"
             rgb1="0.85 0.85 0.85" rgb2="0.75 0.75 0.75"/>
    <material name="floor" texture="grid" texrepeat="8 8"/>
  </asset>
  <worldbody>
    <light pos="2 0 4" dir="0 0 -1" diffuse="0.9 0.9 0.9"/>
    <camera name="demo" pos="3.0 -4.2 4.2" xyaxes="1 0 0 0 0.72 0.72"/>
    <!-- измерительная арка RGB-D над зоной камеры (x = 1.0):
         верхняя + две боковые 45 град, как в промышленных DWS-тоннелях -->
    <camera name="cam_top"   pos="1.0  0.0  2.30" xyaxes="1 0 0 0 1 0" fovy="35"/>
    <camera name="cam_sideL" pos="1.0  1.15 2.55" xyaxes="1 0 0 0 0.847 -0.532" fovy="40"/>
    <camera name="cam_sideR" pos="1.0 -1.15 2.55" xyaxes="1 0 0 0 0.847 0.532" fovy="40"/>
    <geom name="floor" type="plane" size="8 6 0.1" material="floor"
          friction="0.6 0.01 0.003"/>

    <!-- лента: статичный короб, движение имитируется контроллером -->
    <geom name="belt" type="box"
          pos="{BELT_LEN / 2:.3f} 0 {BELT_H - 0.025:.3f}"
          size="{BELT_LEN / 2:.3f} {BELT_W / 2:.3f} 0.025"
          rgba="0.25 0.25 0.3 1" friction="0.5 0.005 0.0001"/>
    <!-- высокие борта на разгонном участке (высокий круглый товар
         не может скатиться); последние 0.4 м — низкие, под поворот лотка -->
    <geom type="box" pos="1.3 {BELT_W / 2 + 0.02:.3f} 0.835"
          size="1.3 0.02 0.215" rgba="0.4 0.4 0.45 1"/>
    <geom type="box" pos="1.3 {-BELT_W / 2 - 0.02:.3f} 0.835"
          size="1.3 0.02 0.215" rgba="0.4 0.4 0.45 1"/>
    <!-- борта ленты (заканчиваются до лотка — зазор для его поворота) -->
    <geom type="box" pos="1.49 {BELT_W / 2 + 0.02:.3f} {BELT_H + 0.05:.3f}"
          size="1.49 0.02 0.08" rgba="0.4 0.4 0.45 1"/>
    <geom type="box" pos="1.49 {-BELT_W / 2 - 0.02:.3f} {BELT_H + 0.05:.3f}"
          size="1.49 0.02 0.08" rgba="0.4 0.4 0.45 1"/>

    <!-- мостик-переходник с ЩЕЛЕВЫМ ОТСЕВОМ: калиброванная щель 40 мм
         сразу за срезом ленты — пассивный габаритный калибр: товар
         тоньше минимального габарита проваливается в реджект-карман
         категории C до накопителя; остальной товар мостит щель.
         Лоток вращается под мостиком -->
    <geom name="bridge" type="box" pos="3.070 0 0.657" euler="0 15 0"
          size="0.030 0.24 0.005" rgba="0.45 0.45 0.5 1"
          friction="0.25 0.005 0.0001"/>
    <!-- реджект-карман под щелью (маркер зоны C) -->
    <geom name="rejectC" type="box" pos="3.15 0 0.005" size="0.30 0.35 0.005"
          rgba="0.9 0.7 0.1 0.35" contype="0" conaffinity="0"/>

    <!-- накопитель-лоток (tilt-tray): yaw поворачивает на зону,
         tilt принимает (контруклон) и опрокидывает -->
    <body name="chute" pos="{X_CHUTE:.3f} 0 {TRAY_Z:.3f}">
      <joint name="chute_yaw" type="hinge" axis="0 0 1"
             range="-75 75" damping="8"/>
      <joint name="chute_tilt" type="hinge" axis="0 1 0"
             range="-6 40" damping="5"/>
      {chute_geoms}
      <!-- торцевой флап: закрыт при приеме (ловит катящийся товар),
           открывается при опрокидывании как продолжение склона -->
      <body name="flap" pos="1.085 0 0.186" euler="0 -18 0">
        <joint name="flap_hinge" type="hinge" axis="0 1 0"
               range="-5 100" damping="1"/>
        <geom type="box" pos="0.012 0 0.055" size="0.012 0.34 0.070"
              rgba="0.9 0.45 0.15 1" density="300"
              friction="0.25 0.005 0.0001"/>
      </body>
    </body>

    <!-- визуальные маркеры зон (contype=0 -> без коллизий) -->
    <geom name="zoneB" type="box" pos="5.0 0 0.01" size="0.9 0.7 0.01"
          rgba="0.2 0.8 0.2 0.35" contype="0" conaffinity="0"/>
    <geom name="zoneC" type="box" pos="4.0 1.9 0.01" size="0.8 0.8 0.01"
          rgba="0.9 0.7 0.1 0.35" contype="0" conaffinity="0"/>
    <geom name="zoneD" type="box" pos="4.0 -1.9 0.01" size="0.8 0.8 0.01"
          rgba="0.8 0.2 0.2 0.35" contype="0" conaffinity="0"/>

    <!-- накопители зон: стенки ловят товар (аналог ролл-кейджей C/D
         и приемного участка B); открытая сторона обращена к лотку -->
    <!-- направляющие юбки: канал от лотка к кейджам C/D
         (вне радиуса поворота лотка) и коридор B -->
    <geom type="box" pos="3.85 0.50 0.11" euler="0 0 0"
          size="0.30 0.02 0.11" rgba="0.55 0.55 0.6 1"/>
    <geom type="box" pos="3.85 -0.50 0.11" euler="0 0 0"
          size="0.30 0.02 0.11" rgba="0.55 0.55 0.6 1"/>
    <geom type="box" pos="4.19 0.64 0.11" euler="0 0 38"
          size="0.28 0.02 0.11" rgba="0.55 0.55 0.6 1"/>
    <geom type="box" pos="3.59 1.38 0.11" euler="0 0 38"
          size="0.28 0.02 0.11" rgba="0.55 0.55 0.6 1"/>
    <geom type="box" pos="4.19 -0.64 0.11" euler="0 0 -38"
          size="0.28 0.02 0.11" rgba="0.55 0.55 0.6 1"/>
    <geom type="box" pos="3.59 -1.38 0.11" euler="0 0 -38"
          size="0.28 0.02 0.11" rgba="0.55 0.55 0.6 1"/>

    {bin_walls('binB', 5.0, 0.0, 0.9, 0.7, 'x-')}
    {bin_walls('binC', 4.0, 1.9, 0.8, 0.8, 'y-')}
    {bin_walls('binD', 4.0, -1.9, 0.8, 0.8, 'y+')}

    {''.join(bodies)}
  </worldbody>
  <actuator>
    <position name="servo_yaw" joint="chute_yaw" kp="600" kv="60"
              ctrlrange="-1.32 1.32"/>
    <position name="servo_tilt" joint="chute_tilt" kp="2500" kv="180"
              ctrlrange="-0.10 0.70"/>
    <position name="servo_flap" joint="flap_hinge" kp="80" kv="8"
              ctrlrange="-0.05 1.60"/>
  </actuator>
</mujoco>'''


ZONES = [  # (имя, x_min, x_max, y_min, y_max), проверка по порядку.
    # Реджект-карман отсева: под мостиком И под лотком (микро-габарит,
    # не сошедший в кейдж, оседает в кармане) — категория C
    ('C', 2.85, 3.75, -0.40, 0.40),
    ('C', 3.05, 4.90, 0.62, 2.70),
    ('D', 3.05, 4.90, -2.70, -0.62),
    ('B', 3.82, 5.90, -0.68, 0.68),
]


def zone_of(x: float, y: float) -> str:
    for name, x0, x1, y0, y1 in ZONES:
        if x0 <= x <= x1 and y0 <= y <= y1:
            return name
    return '?'


def run(video: bool = False, variants=None,
        log_name: str = 'sim_log.csv') -> list[dict]:
    objects = load_categories()
    if variants is None:
        variants = [(i, 0.0, 0.0) for i in range(len(objects))]
    model = mujoco.MjModel.from_xml_string(build_mjcf(objects))
    data = mujoco.MjData(model)
    a_yaw = model.actuator('servo_yaw').id
    a_tilt = model.actuator('servo_tilt').id
    a_flap = model.actuator('servo_flap').id
    q_yaw = model.joint('chute_yaw').qposadr[0]
    receive = np.radians(RECEIVE_TILT)
    dump = np.radians(DUMP_TILT)

    renderer = frames = None
    if video:
        renderer = mujoco.Renderer(model, height=720, width=1280)
        frames = []

    belt_gid = model.geom('belt').id
    halfz = half_heights(objects)
    warn_id = mujoco.mjtWarning.mjWARN_BADQACC

    def on_belt(gid):
        for k in range(data.ncon):
            c = data.contact[k]
            if {c.geom1, c.geom2} & {belt_gid} and gid in (c.geom1, c.geom2):
                return True
        return False

    log, events = [], []
    for i, y_off, yaw_deg in variants:
        row = objects[i]
        jadr = model.joint(f'fj{i}').qposadr[0]
        vadr = model.joint(f'fj{i}').dofadr[0]
        bid = model.body(f'obj{i}').id
        obj_gid = model.body(f'obj{i}').geomadr[0]
        mass = float(model.body_mass[bid])
        # подача товара на ленту (мягко, по габариту); лоток в приём
        n_warn0 = int(data.warning[warn_id].number)
        half_ang = np.radians(yaw_deg) / 2
        data.qpos[jadr:jadr + 7] = [X_SPAWN, y_off,
                                    BELT_H + halfz[i] + 0.02,
                                    np.cos(half_ang), 0, 0, np.sin(half_ang)]
        data.qvel[vadr:vadr + 6] = 0
        data.ctrl[a_yaw] = 0.0
        data.ctrl[a_tilt] = receive
        data.ctrl[a_flap] = 0.0
        mujoco.mj_forward(model, data)

        t0 = data.time
        category = row['category']
        target_yaw = np.radians(YAW[category])
        deadline = t0 + 30.0

        t_classified = t_eta = t_zone = t_dump = None
        phase = 'belt'          # belt -> rotate -> dump -> done
        boosted = False
        result_zone = None
        jam_logged = False

        while data.time < deadline:
            x, y, z = data.qpos[jadr:jadr + 3]
            speed = float(np.linalg.norm(data.qvel[vadr:vadr + 3]))

            # --- лента: тянет товар, ПОКА он касается полотна (как
            # реальная лента); последние 0.4 м — пассивный тормозной
            # фартук гасит вход в лоток до ~0.45 м/с
            touching_belt = on_belt(obj_gid)
            if touching_belt and data.time > t0 + 0.4:
                data.qvel[vadr] = BELT_V if x < 2.6 else 0.45
                # центрирование к оси ленты (направляющие скирты)
                data.qvel[vadr + 1] = float(np.clip(-3.0 * y, -0.4, 0.4))
                # распределенный контакт ленты гасит вращение товара
                data.qvel[vadr + 3:vadr + 6] *= 0.85

            # --- CV-модуль: товар в зоне камеры -> классификация ---
            if t_classified is None and x >= X_CAMERA:
                t_classified = data.time + INFER_TIME
                # синхронизация: расчётное время прибытия товара в лоток
                t_eta = data.time + (X_CHUTE + 0.3 - X_CAMERA) / BELT_V
                if row['manual_review_required'] == 'True':
                    events.append({'t': round(data.time, 2), 'obj': row['name'],
                                   'event': 'manual_review_required',
                                   'detail': row['reason']})

            # --- конечный автомат лотка ---
            if phase == 'belt':
                landed = (not touching_belt and x > X_CHUTE - 0.03
                          and z > 0.35 and speed < 0.35)
                # ПЛК-таймаут: товар в лотке, но критерий покоя не сработал
                # (микроколебания) — продолжаем цикл принудительно
                if (not landed and t_eta is not None
                        and data.time > t_eta + 3.0
                        and not touching_belt and x > X_CHUTE - 0.03
                        and z > 0.35):
                    landed = True
                    events.append({'t': round(data.time, 2),
                                   'obj': row['name'],
                                   'event': 'landed_forced',
                                   'detail': 'приём подтвержден по таймауту'})
                if (landed and t_classified is not None
                        and data.time >= t_classified):
                    phase = 'rotate'
                    data.ctrl[a_yaw] = target_yaw
                elif (t_eta is not None and data.time > t_eta + JAM_RESERVE
                        and not jam_logged and z > 0.4):
                    events.append({'t': round(data.time, 2), 'obj': row['name'],
                                   'event': 'jam_suspected',
                                   'detail': 'товар не прибыл в лоток к '
                                             'расчетному времени'})
                    jam_logged = True
            elif phase == 'rotate':
                if abs(data.qpos[q_yaw] - target_yaw) < np.radians(3):
                    phase = 'dump'
                    t_dump = data.time
                    data.ctrl[a_tilt] = dump
                    data.ctrl[a_flap] = np.radians(78)
            elif phase == 'dump':
                # дожим: товар завис на лотке — наклон до максимума
                on_tray = (np.hypot(x - X_CHUTE, y) < 1.25
                           and z > 0.15)
                if (t_dump is not None and data.time > t_dump + 2.5
                        and on_tray and not boosted):
                    data.ctrl[a_tilt] = np.radians(38.5)
                    boosted = True
                    events.append({'t': round(data.time, 2),
                                   'obj': row['name'],
                                   'event': 'dump_boost',
                                   'detail': 'товар завис — наклон до 36.5°'})
                if z < 0.35 and x > BELT_LEN:
                    zn = zone_of(x, y)
                    if zn != '?' and t_zone is None:
                        t_zone = data.time
                if t_zone is not None and data.time > t_zone + SETTLE_TIME:
                    result_zone = zone_of(data.qpos[jadr],
                                          data.qpos[jadr + 1])
                    break

            mujoco.mj_step(model, data)
            if abs(data.qpos[jadr + 1]) > 4.0 or not (-2 < data.qpos[jadr] < 8):
                events.append({'t': round(data.time, 2), 'obj': row['name'],
                               'event': 'ejected_estop',
                               'detail': 'товар вылетел за пределы участка'})
                break
            if int(data.warning[warn_id].number) > n_warn0:
                events.append({'t': round(data.time, 2), 'obj': row['name'],
                               'event': 'unstable_contact_estop',
                               'detail': 'BADQACC: авто-сброс решателя, '
                                         'объект снят с линии'})
                break
            if np.isnan(data.qpos[jadr:jadr + 7]).any():
                # контактная неустойчивость: аварийный съём объекта с линии
                events.append({'t': round(data.time, 2), 'obj': row['name'],
                               'event': 'unstable_contact_estop',
                               'detail': 'NaN в состоянии, объект снят с линии'})
                data.qpos[jadr:jadr + 7] = [-2 - i * 0.8, 3.0, 0.3, 1, 0, 0, 0]
                data.qvel[vadr:vadr + 6] = 0
                mujoco.mj_forward(model, data)
                break
            if renderer is not None and int(round(data.time / DT)) % 16 == 0:
                renderer.update_scene(data, camera='demo')
                frames.append(renderer.render())

        if result_zone is None:
            result_zone = zone_of(data.qpos[jadr], data.qpos[jadr + 1])
            if result_zone == '?':
                events.append({'t': round(data.time, 2), 'obj': row['name'],
                               'event': 'stuck_or_lost',
                               'detail': 'товар не достиг зоны за таймаут'})

        cycle = (t_zone - t0) if t_zone else None
        log.append({
            'name': row['name'], 'category': category,
            'spawn_y': y_off, 'spawn_yaw': yaw_deg,
            'target_zone': category, 'actual_zone': result_zone,
            'correct': result_zone == category,
            'final_x': round(float(data.qpos[jadr]), 2),
            'final_y': round(float(data.qpos[jadr + 1]), 2),
            't_classify_s': round((t_classified - t0), 2) if t_classified else '',
            't_zone_entry_s': round(cycle, 2) if cycle else '',
            'manual_review': row['manual_review_required'],
        })
        print(f"{row['name']:<22} {category} -> {result_zone}  "
              f"{'OK' if result_zone == category else 'FAIL'}  "
              f"cycle={cycle and round(cycle, 2)}s")

        # паркуем отработанный товар; лоток возвращается в приём
        data.xfrc_applied[bid, :] = 0.0
        data.qpos[jadr:jadr + 7] = [-2 - i * 0.8, 3.0, 0.3, 1, 0, 0, 0]
        data.qvel[vadr:vadr + 6] = 0
        data.ctrl[a_yaw] = 0.0
        data.ctrl[a_tilt] = receive
        data.ctrl[a_flap] = 0.0
        t_pause = data.time + 1.2
        while data.time < t_pause:
            mujoco.mj_step(model, data)

    os.makedirs(os.path.join(ROOT, 'output'), exist_ok=True)
    with io.open(os.path.join(ROOT, 'output', log_name), 'w',
                 encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader(); w.writerows(log)
    if events:
        with io.open(os.path.join(ROOT, 'output', 'sim_events.csv'), 'w',
                     encoding='utf-8-sig', newline='') as f:
            w = csv.DictWriter(f, fieldnames=['t', 'obj', 'event', 'detail'])
            w.writeheader(); w.writerows(events)

    ok = sum(1 for r in log if r['correct'])
    cycles = [r['t_zone_entry_s'] for r in log if r['t_zone_entry_s'] != '']
    print(f'\nМаршрутизация: {ok}/{len(log)} корректно.')
    if cycles:
        print(f'Cycle time: среднее {np.mean(cycles):.2f} с, '
              f'макс {np.max(cycles):.2f} с')
    print('Лог: output/sim_log.csv')

    if frames:
        try:
            import imageio
            imageio.mimsave(os.path.join(ROOT, 'output', 'sim_demo.mp4'),
                            frames, fps=30)
            print('Видео: output/sim_demo.mp4')
        except ImportError:
            print('imageio не установлен — видео пропущено')
    return log


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', action='store_true')
    ap.add_argument('--stress', action='store_true',
                    help='вариации подачи: смещение и разворот товара')
    args = ap.parse_args()
    if args.stress:
        import trimesh
        objs = load_categories()
        variants = []
        for i, row in enumerate(objs):
            stl = os.path.join(ROOT, 'data', 'models_sim', row['file'])
            b = trimesh.load(stl, force='mesh').bounds * 1e-3
            half_xy = float(max(b[1][0] - b[0][0], b[1][1] - b[0][1])) / 2
            dy = max(0.0, min(0.10, 0.22 - half_xy))
            variants += [(i, 0.0, 0.0), (i, +dy, 25.0), (i, -dy, -40.0)]
        run(variants=variants, log_name='stress_log.csv')
    else:
        run(video=args.video)
