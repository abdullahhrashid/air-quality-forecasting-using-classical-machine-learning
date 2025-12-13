#this script too, is from gemini, this stuff is some bull that i refuse to learn out of spite
import xarray as xr
import pandas as pd
import os

# 1. Setup Path
path = os.path.join(os.path.dirname(__file__), '../../data/raw/solar')
file_pattern = os.path.join(path, '*.nc')

# 2. Load Data
ds = xr.open_mfdataset(file_pattern, combine='by_coords')

# --- DEBUG PRINT ---
print("Variables found in file:", list(ds.data_vars))
# -------------------

# 3. Handle expver (if present)
if 'expver' in ds.dims:
    ds = ds.mean(dim='expver', skipna=True)

# 4. Rename variables to be human-readable
# We use a try/except block just in case the name is slightly different
rename_dict = {
    'ssr': 'solar_radiation',
    'blh': 'boundary_layer_height'
}
# Only rename columns that actually exist
existing_vars = list(ds.data_vars)
to_rename = {k: v for k, v in rename_dict.items() if k in existing_vars}
ds = ds.rename(to_rename)

# 5. Spatial Average (Collapse Grid to Single Point)
ds_delhi = ds.mean(dim=['latitude', 'longitude'])

# 6. Daily Average (Resample)
ds_daily = ds_delhi.resample(valid_time='1D').mean()

# 7. Convert Units (Joules -> Watts)
# ERA5 solar radiation is accumulated over the hour (3600s).
# Watts = Joules / Seconds
if 'solar_radiation' in ds_daily:
    ds_daily['solar_radiation'] = ds_daily['solar_radiation'] / 3600

# 8. Convert to DataFrame and Clean
df = ds_daily.to_dataframe()

# Drop useless columns
cols_to_drop = ['number', 'expver']
df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

# 9. Save
output_path = os.path.join(path, '../../processed/solar.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path)

print("\n--- FINAL DATAFRAME HEAD ---")
print(df.head())
print(f"\nSaved to: {output_path}")