# -*- coding: utf-8 -*-
"""
Мост «часть классификации → физический узел».

Берёт результат классификации (output/classification.csv или переданную
категорию) и отправляет команду на контроллер лотка ESP8266 по HTTP —
та же величина B/C/D, что уходит в цифровой двойник. Этим замыкается
связка CV ↔ исполнительная часть на реальном железе.

Примеры:
    # одна категория
    python hardware/send_category.py --host 192.168.4.1 --cat D

    # прогнать весь тестовый набор из classification.csv по очереди
    python hardware/send_category.py --host 192.168.4.1 --from-csv

ESP8266 в режиме точки доступа обычно доступен по адресу 192.168.4.1.
"""
import argparse
import csv
import io
import os
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def send(host: str, cat: str, timeout: float = 5.0) -> str:
    url = f"http://{host}/sort?cat={cat}"
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read().decode(errors="replace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="192.168.4.1", help="IP контроллера ESP8266")
    ap.add_argument("--cat", choices=["B", "C", "D"], help="одна категория")
    ap.add_argument("--from-csv", action="store_true",
                    help="прогнать все товары из output/classification.csv")
    ap.add_argument("--delay", type=float, default=5.0,
                    help="пауза между товарами, с")
    args = ap.parse_args()

    if args.from_csv:
        path = os.path.join(ROOT, "output", "classification.csv")
        with io.open(path, encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                cat = row["category"]
                print(f'{row["name"]:<22} -> {cat}: {send(args.host, cat)}')
                time.sleep(args.delay)
    elif args.cat:
        print(send(args.host, args.cat))
    else:
        ap.error("укажите --cat B|C|D или --from-csv")


if __name__ == "__main__":
    main()
