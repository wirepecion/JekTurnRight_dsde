import re
import time
from datetime import datetime, timedelta
from typing import List
import requests
import pandas as pd

URL = "https://weather.bangkok.go.th/rain/RainHistory/ContourRain"
HEADERS = {"Content-Type": "application/x-www-form-urlencoded"}
PATTERN = re.compile(r"[-\d\.]+\s+[-\d\.]+\s+([\d\.\-]+|\s*)\s*RF\.(\S+)")

def generate_payload(date_str: str) -> dict:
    is_dec_31 = date_str.startswith("31/12")
    payload = {
        "datePick_now": date_str,
        "datePick_now0to24": date_str,
        "datePick_start": date_str,
        "StationTime_start": "00:00",
        "datePick_end": date_str,
        "StationTime_end": "23:59",
        "button": "print",
        "note": "imsorrythisisformyprojectindatasciencethereisnointtentiontoattack",
    }
    if is_dec_31:
        payload["rain_type"] = "10"
        payload["StationTime_end"] = "23:55"
    else:
        payload["rain_type"] = "1"
    return payload


def build_date_range(start_date: str, end_date: str) -> List[str]:
    start = datetime.strptime(start_date, "%d/%m/%Y")
    end = datetime.strptime(end_date, "%d/%m/%Y")
    dates = []
    cur = start
    while cur <= end:
        dates.append(cur.strftime("%d/%m/%Y"))
        cur += timedelta(days=1)
    return dates


def scrape_one_day(date_str: str, session: requests.Session | None = None) -> pd.DataFrame | None:
    sess = session or requests.Session()
    payload = generate_payload(date_str)

    try:
        resp = sess.post(URL, data=payload, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"[{date_str}] Request error: {e}")
        return None

    records = []
    for line in resp.text.splitlines():
        m = PATTERN.search(line)
        if not m:
            continue
        water_level_str = m.group(1).strip()
        station = m.group(2)

        if not water_level_str:
            water_level = None
        else:
            try:
                water_level = float(water_level_str)
            except ValueError:
                water_level = None

        records.append((date_str, station, water_level))

    if not records:
        print(f"[{date_str}] No records found.")
        return None

    df = pd.DataFrame(records, columns=["Date", "Station", "WaterLevel"])
    return df


def scrape_range(start_date: str, end_date: str, sleep_sec: float = 0.5) -> pd.DataFrame:
    dates = build_date_range(start_date, end_date)
    print(f"Starting scrape for dates: {', '.join(dates)}")
    all_rows = []
    start_time = time.time()

    with requests.Session() as session:
        for d in dates:
            print(f"Processing date: {d}...")
            df = scrape_one_day(d, session=session)
            if df is not None:
                all_rows.append(df)
            time.sleep(sleep_sec)  # be nice

    if not all_rows:
        raise RuntimeError("No data scraped for any date.")

    long_df = pd.concat(all_rows, ignore_index=True)
    # Pivot once at the end
    wide = long_df.pivot_table(
        index="Date", columns="Station", values="WaterLevel", aggfunc="first"
    ).reset_index()

    elapsed = time.time() - start_time
    print(f"Done. {len(wide)} days. Took {elapsed:.2f}s.")
    return wide
