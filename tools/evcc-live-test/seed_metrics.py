"""Seed evcc's metrics DB with 31 days of 15-minute energy history.

The optimizer needs a 30-day home profile (`site.homeProfile`); a fresh evcc
aborts with `optimizer: meter profile incomplete` long before it builds a
request. This is the actual obstacle to an end-to-end test - not the
sponsorship gate.

    python seed_metrics.py /path/to/evcc.db

Entity ids come from the `entities` table, which evcc creates on first run, so
start evcc once before seeding.
"""

import math
import sqlite3
import sys
import time

SLOT = 900  # seconds


def main(path: str) -> None:
    db = sqlite3.connect(path)
    ents = dict(db.execute("select name, id from entities").fetchall())
    home = ents.get("home")
    if home is None:
        raise SystemExit("no 'home' entity - start evcc once first")

    now = int(time.time())
    start = (now // SLOT) * SLOT - 31 * 96 * SLOT
    rows = []
    for i in range(31 * 96 + 4):
        ts = start + i * SLOT
        hod = (ts % 86400) / 3600.0
        # kWh per 15-min slot: overnight base plus morning and evening bumps
        kwh = (
            0.10
            + 0.25 * math.exp(-0.5 * ((hod - 7.5) / 1.2) ** 2)
            + 0.30 * math.exp(-0.5 * ((hod - 19.5) / 1.8) ** 2)
        )
        rows.append((home, ts, round(kwh, 5), 0.0, 0.0, 0))

        solar = max(0.0, math.sin((hod - 6) / 12 * math.pi)) if 6 < hod < 18 else 0.0
        for name in ("pv1", "forecast"):
            if name in ents:
                rows.append((ents[name], ts, round(solar, 5), 0.0, 0.0, 0))

    db.executemany(
        "insert or replace into meters (meter, ts, energy, return_energy, soc_temp, recovered)"
        " values (?,?,?,?,?,?)",
        rows,
    )
    db.commit()
    print(f"seeded {len(rows)} rows into {path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "evcc.db")
