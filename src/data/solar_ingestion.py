#this script is from gemini btw, i refuse to learn this xarray stuff
import cdsapi
import os
import zipfile
import xarray as xr
import shutil

# Setup directories
save_dir = r'data/raw/solar'
temp_dir = r'data/raw/temp_extract'
os.makedirs(save_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

c = cdsapi.Client()

for year in range(2019, 2026):
    print(f"\nProcessing {year}...")
    zip_path = os.path.join(save_dir, f'{year}_raw.zip')
    final_nc_path = os.path.join(save_dir, f'{year}.nc')
    
    # 1. Download to a ZIP file
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'data_format': 'netcdf',
                'product_type': 'reanalysis',
                'variable': [
                    'boundary_layer_height',
                    'surface_net_solar_radiation',
                ],
                'year': str(year),
                'month': [f"{m:02d}" for m in range(1, 13)],
                'day': [f"{d:02d}" for d in range(1, 32)],
                'time': ['00:00', '06:00', '12:00', '18:00'],
                'area': [29.0, 76.8, 28.2, 77.6],
            },
            zip_path
        )
    except Exception as e:
        print(f"Download failed for {year}: {e}")
        continue

    # 2. Extract ALL files from the zip
    print(f"Extracting {year}...")
    # Clear temp folder first
    for f in os.listdir(temp_dir):
        os.remove(os.path.join(temp_dir, f))
        
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
            print(f"   Found files: {zip_ref.namelist()}")
    except zipfile.BadZipFile:
        # Sometimes CDS sends a single NC file masquerading as a zip if there's only 1 var
        # In that case, we just rename the download
        print("   Not a zip file (single file returned). Renaming directly.")
        shutil.move(zip_path, final_nc_path)
        continue

    # 3. Merge the split files (BLH + Solar) into one
    try:
        # Load all extracted NetCDF files
        nc_files = [os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith('.nc')]
        
        if nc_files:
            # Open and merge them
            ds = xr.open_mfdataset(nc_files, combine='by_coords')
            ds.to_netcdf(final_nc_path)
            ds.close()
            print(f"   Success! Merged {len(nc_files)} files into {final_nc_path}")
        else:
            print("   Error: No .nc files found in the zip!")
            
    except Exception as e:
        print(f"   Error merging files: {e}")

    # 4. Cleanup
    if os.path.exists(zip_path):
        os.remove(zip_path)

print("\nAll downloads finished. Now run your aggregation script again.")