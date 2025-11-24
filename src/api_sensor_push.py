import time
import argparse
import json
import logging
import random
from typing import Dict

import requests


DEFAULT_BODY = {
    "id_sensor": 1,
    "data": {
        "Suhu_Udara": 25.3,
        "Kelembaban_Udara": 60,
        "Kelembaban_Tanah": 47,
        "pH_Tanah": 6.4,
        "Intensitas_Cahaya": 10460,
    },
}


def jitter_value(value, frac=0.02):
    """Return value jittered +/- frac (fractional) for numeric values."""
    try:
        v = float(value)
    except Exception:
        return value
    delta = abs(v) * frac
    return v + random.uniform(-delta, delta)


def build_body(base: Dict, randomize: bool = False, randomize_id: bool = False, id_min: int = 1, id_max: int = 10) -> Dict:
    """Build payload. If `randomize` is True, jitter numeric fields.
    If `randomize_id` is True, pick a random integer id between `id_min` and `id_max`.
    """
    # decide id_sensor
    if randomize_id:
        id_sensor = random.randint(id_min, id_max)
    else:
        id_sensor = base.get("id_sensor", 1)

    b = {"id_sensor": id_sensor, "data": {}}
    for k, v in base.get("data", {}).items():
        b["data"][k] = jitter_value(v, frac=0.03) if randomize else v
    return b


def push_loop(url: str, body: Dict, interval: float, randomize: bool, randomize_id: bool = False,
              id_min: int = 1, id_max: int = 10, headers: Dict = None):
    headers = headers or {"Content-Type": "application/json"}
    logging.info("Start pushing to %s every %.2f seconds", url, interval)
    try:
        while True:
            payload = build_body(body, randomize=randomize, randomize_id=randomize_id, id_min=id_min, id_max=id_max)
            try:
                resp = requests.post(url, json=payload, headers=headers, timeout=10)
                logging.info("POST %s -> %s (status=%s)", url, json.dumps(payload), resp.status_code)
            except Exception as e:
                logging.warning("Request failed: %s", e)
            time.sleep(interval)
    except KeyboardInterrupt:
        logging.info("Interrupted by user, stopping push loop")


def main():
    parser = argparse.ArgumentParser(description="Send sensor JSON to API repeatedly")
    parser.add_argument("--url", type=str, default="http://127.0.0.1:8000/api/sensor/firebase",
                        help="Destination URL to POST the JSON payload")
    parser.add_argument("--id", type=int, default=1, help="Sensor id to send in payload")
    parser.add_argument("--interval", type=float, default=3.0, help="Seconds between sends (default 3)")
    parser.add_argument("--randomize", action="store_true", help="Slightly randomize numeric values each send")
    parser.add_argument("--random-id", action="store_true", help="Randomize `id_sensor` each send")
    parser.add_argument("--id-min", type=int, default=1, help="Minimum random sensor id (inclusive)")
    parser.add_argument("--id-max", type=int, default=10, help="Maximum random sensor id (inclusive)")
    parser.add_argument("--suhu", type=float, default=None, help="Override Suhu_Udara")
    parser.add_argument("--kelembaban_udara", type=float, default=None, help="Override Kelembaban_Udara")
    parser.add_argument("--kelembaban_tanah", type=float, default=None, help="Override Kelembaban_Tanah")
    parser.add_argument("--ph", type=float, default=None, help="Override pH_Tanah")
    parser.add_argument("--cahaya", type=float, default=None, help="Override Intensitas_Cahaya")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    body = DEFAULT_BODY.copy()
    body["id_sensor"] = args.id
    # copy data dict
    data = body["data"].copy()
    if args.suhu is not None:
        data["Suhu_Udara"] = args.suhu
    if args.kelembaban_udara is not None:
        data["Kelembaban_Udara"] = args.kelembaban_udara
    if args.kelembaban_tanah is not None:
        data["Kelembaban_Tanah"] = args.kelembaban_tanah
    if args.ph is not None:
        data["pH_Tanah"] = args.ph
    if args.cahaya is not None:
        data["Intensitas_Cahaya"] = args.cahaya
    body["data"] = data

    push_loop(args.url, body, interval=args.interval, randomize=args.randomize,
              randomize_id=args.random_id, id_min=args.id_min, id_max=args.id_max)


if __name__ == "__main__":
    main()
