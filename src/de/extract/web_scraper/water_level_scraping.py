import requests
import pandas as pd
import re
from datetime import datetime, timedelta
import time
# Use a function to simplify generating the payload based on the date
def generate_payload(d):
    """Generates the appropriate payload for a given date."""
    is_dec_31 = d.startswith('31/12')

    # Base payload for all dates
    payload = {
        "datePick_now": d,
        "datePick_now0to24": d,
        "datePick_start": d,
        "StationTime_start": "00:00",
        "datePick_end": d,
        "StationTime_end": "23:59",
        "button": "print",
        "note": "imsorrythisisformyprojectindatasciencethereisnointtentiontoattack"
    }

    if is_dec_31:
        # Special case for December 31st (using 23:55 end time and rain_type=10)
        # Note: I'll use 23:59 for consistency unless 23:55 is strictly necessary for the API.
        # Since your example URL uses 23:55, I'll adhere to that for the special case.
        payload["rain_type"] = "10"
        payload["StationTime_end"] = "23:55"
        # The example you provided uses datePick_now='18/11/2025' which seems inconsistent
        # for a Dec 31st scrape, so I'll keep all dates consistent for the actual day being scraped.
    else:
        # Normal case for all other dates
        payload["rain_type"] = "1"
        payload["StationTime_end"] = "23:59"
    return payload

# --- Configuration ---
# --- Please carefully check these configuration line ----
start_date_str = "01/01/2022"  # Start scraping date
end_date_str = "10/01/2022"
# --- Please carefully check these configuration line ----
output_file_name = "rain_water_"+ start_date_str.replace('/', '-')+"_to_"+end_date_str.replace('/', '-')+".csv"
start_date = datetime.strptime(start_date_str, "%d/%m/%Y")
end_date = datetime.strptime(end_date_str, "%d/%m/%Y")

dates = []
current_date = start_date
while current_date <= end_date:
    dates.append(current_date.strftime("%d/%m/%Y"))
    current_date += timedelta(days=1)

# Overwrite for brief testing
#dates = ["01/01/2022", "18/11/2021", "20/11/2021"] 
# --- End of Date Configuration ---

url = "https://weather.bangkok.go.th/rain/RainHistory/ContourRain"
headers = {"Content-Type": "application/x-www-form-urlencoded"}

# Regex to extract water level + station code
# NEW PATTERN: 
# 1. 'X.XX Y.YY Z.ZZ RF.ABC' -> captures Z.ZZ
# 2. 'X.XX Y.YY         RF.ABC' -> captures the blank space as the rain level
pattern = re.compile(r'[-\d\.]+\s+[-\d\.]+\s+([\d\.\-]+|\s*)\s*RF\.(\S+)')

all_dfs = []

print(f"Starting scrape for dates: {', '.join(dates)}")
print("---")

### TIMING START ###
start_time = time.time() 
### TIMING START ###
for d in dates:
    print(f"Processing date: {d}...")
    payload = generate_payload(d)

    try:
        response = requests.post(url, data=payload, headers=headers, timeout=30)
        response.raise_for_status() 
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data for {d}: {e}")
        continue

    raw_text = response.text
    records = []
    
    for line in raw_text.splitlines():
        match = pattern.search(line)
        if match:
            # Group 1 is the rain level, Group 2 is the station code
            water_level_str = match.group(1).strip()
            station = match.group(2)
            
            # Case 1: Station exists but rain value is blank/missing
            if not water_level_str:
                water_level = None
            # Case 2: Station and rain value exist
            else:
                try:
                    water_level = float(water_level_str)
                except ValueError:
                    # Handle cases where the captured string might not be a valid float
                    water_level = None 
            
            records.append((station, water_level))

    if records:
        df = pd.DataFrame(records, columns=['Station', 'WaterLevel'])
        df['Date'] = d
        
        # Stations present on other dates but not this one will have NaN.
        df_pivot = df.pivot(index='Date', columns='Station', values='WaterLevel').reset_index()
        all_dfs.append(df_pivot)
    else:
        print(f"No rain records found for {d}.")

# Combine all dates
# This step handles the 'non-existent station' requirement.
# pd.concat aligns columns by name. If a column (Station) is missing in one DataFrame 
# but present in others, it fills the missing spots with NaN (null).
final_df = pd.concat(all_dfs, ignore_index=True)

# Reorder columns: 'Date' first, then all station columns alphabetically
cols = ['Date'] + sorted([col for col in final_df.columns if col != 'Date'])
final_df = final_df[cols]

# Save to CSV
final_df.to_csv(output_file_name, index=False, encoding="utf-8-sig")

### TIMING END ###
end_time = time.time()
total_time_seconds = end_time - start_time
total_time_minutes = total_time_seconds / 60
### TIMING END ###

print("---")
print("✅ Successfully scarping data.")
print(f"Saved merged rain data to **{output_file_name}** containing {len(final_df)} days of data.")
print(f"⏱️ Total time taken: **{total_time_seconds:.2f} seconds** ({total_time_minutes:.2f} minutes)")

'''
curl -X POST "https://weather.bangkok.go.th/rain/RainHistory/ContourRain" ^
-H "Content-Type: application/x-www-form-urlencoded" ^
-d "rain_type=1&datePick_now=17%2F11%2F2025&datePick_now0to24=17%2F11%2F2023&datePick_start=17%2F11%2F2025&StationTime_start=19%3A29&datePick_end=18%2F11%2F2025&StationTime_end=19%3A29&button=print&note=imsorrythisisformyprojectindatasciencethereisnointtentiontoattack"

,
    "note":"imsorrythisisformyprojectindatasciencethereisnointtentiontoattack"
'''