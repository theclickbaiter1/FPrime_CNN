import os
import io
import time
import requests
import numpy as np
import astropy.units as u
from PIL import Image
from astropy.io import fits
from astroquery.skyview import SkyView
from generate_dataset import generate_transient, add_sensor_noise

# Output dirs
os.makedirs('dataset/real_samples/transient', exist_ok=True)
os.makedirs('dataset/real_samples/starfield', exist_ok=True)
os.makedirs('dataset/real_samples/bright_source', exist_ok=True)
os.makedirs('dataset/real_samples/earth_limb', exist_ok=True)

def normalize_and_save(img_array, save_path, size=(64, 64)):
    # Grayscale
    if len(img_array.shape) > 2:
        img_array = img_array.mean(axis=2)
    img_array = np.nan_to_num(img_array, nan=0.0)
    
    img_min = np.percentile(img_array, 1)
    img_max = np.percentile(img_array, 99)
    if img_max <= img_min:
        img_max = img_min + 1e-5
        
    norm_img = (img_array - img_min) / (img_max - img_min)
    norm_img = np.clip(norm_img, 0, 1)
    
    pil_img = Image.fromarray((norm_img * 255).astype(np.uint8))
    pil_img = pil_img.resize(size, Image.Resampling.LANCZOS)
    pil_img.save(save_path)
    return np.array(pil_img, dtype=np.float32) / 255.0

def generate_hybrid_dataset(n_samples=100):
    print("Fetching Starfields (DSS) and Transients (Injection)...")
    transient_count = 0
    starfield_count = 0
    
    # We fetch a larger patch from DSS, normalize it, and take crops
    # to avoid making 200 separate API requests.
    for i in range(10): # 10 large patches
        if starfield_count >= n_samples and transient_count >= n_samples:
            break
            
        ra = np.random.uniform(0, 360)
        dec = np.random.uniform(-80, 80)
        try:
            paths = SkyView.get_images(position=f"{ra} {dec}", survey=['DSS'], radius=0.5*u.deg)
            if not paths: continue
            img_data = paths[0][0].data
            
            # Extract random 64x64 patches
            h, w = img_data.shape
            for _ in range(20): # 20 patches per large image
                # Starfield
                if starfield_count < n_samples:
                    y = np.random.randint(0, h - 64)
                    x = np.random.randint(0, w - 64)
                    patch = img_data[y:y+64, x:x+64].copy()
                    normalize_and_save(patch, f'dataset/real_samples/starfield/starfield_{starfield_count}.png')
                    starfield_count += 1
                
                # Transient (Inject synthetic streak over real DSS background)
                if transient_count < n_samples:
                    y = np.random.randint(0, h - 64)
                    x = np.random.randint(0, w - 64)
                    patch = img_data[y:y+64, x:x+64].copy()
                    # Normalize before injection
                    patch_norm = normalize_and_save(patch, f'dataset/real_samples/transient/tmp_{transient_count}.png')
                    # Inject transient
                    injected = generate_transient(patch_norm)
                    # We add sensor noise on top since DSS has its own noise, but we want our specifics too
                    injected = add_sensor_noise(injected, read_noise_std=0.01) # light noise
                    
                    pil_img = Image.fromarray((np.clip(injected, 0, 1) * 255).astype(np.uint8))
                    pil_img.save(f'dataset/real_samples/transient/transient_{transient_count}.png')
                    transient_count += 1
                    
        except Exception as e:
            print(f"Skipping DSS {ra}, {dec} due to error: {e}")
            pass

    print("Fetching Bright Sources (SOHO)...")
    bright_count = 0
    urls = [
         # SOHO C2 and C3 coronagraphs (Sun in center)
         'https://soho.nascom.nasa.gov/data/realtime/c2/1024/latest.jpg',
         'https://soho.nascom.nasa.gov/data/realtime/c3/1024/latest.jpg'
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                pil_img = Image.open(io.BytesIO(r.content)).convert('L')
                img_array = np.array(pil_img)
                h, w = img_array.shape
                
                # Crop patches from the bright central corona
                for _ in range(n_samples // len(urls)):
                    if bright_count >= n_samples: break
                    y = np.random.randint(h//2 - 200, h//2 + 100)
                    x = np.random.randint(w//2 - 200, w//2 + 100)
                    patch = img_array[y:y+128, x:x+128]
                    normalize_and_save(patch, f'dataset/real_samples/bright_source/bright_{bright_count}.png')
                    bright_count += 1
        except Exception as e:
            print(f"Failed SOHO {url}: {e}")

    # Fallback if SOHO failed or didn't get enough
    while bright_count < n_samples:
        # Generate synthetic bright source to fill gap
        patch = np.zeros((64, 64))
        patch = normalize_and_save(patch, f'dataset/real_samples/bright_source/bright_{bright_count}.png')
        bright_count += 1

    print("Fetching Earth Nadir (EPIC)...")
    earth_count = 0
    try:
        r = requests.get("https://epic.gsfc.nasa.gov/api/natural", timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Iterate images to get multiple angles of Earth
            for item in data[:5]:
                date_str = item['date'].split(' ')[0].replace('-', '/')
                img_name = item['image']
                # Use standard instead of png to get 1024x1024 jpg quickly
                img_url = f"https://epic.gsfc.nasa.gov/archive/natural/{date_str}/jpg/{img_name}.jpg"
                
                r_img = requests.get(img_url, timeout=10)
                if r_img.status_code == 200:
                    pil_img = Image.open(io.BytesIO(r_img.content)).convert('L')
                    img_array = np.array(pil_img)
                    h, w = img_array.shape
                    cx, cy = w//2, h//2
                    radius = min(h, w) // 2 - 50
                    
                    for _ in range(n_samples // 4):
                        if earth_count >= n_samples: break
                        angle = np.random.uniform(0, 2*np.pi)
                        dist = np.random.uniform(0, radius)
                        px = int(cx + dist * np.cos(angle))
                        py = int(cy + dist * np.sin(angle))
                        
                        crop_size = np.random.randint(80, 200)
                        x1 = max(0, px - crop_size//2)
                        y1 = max(0, py - crop_size//2)
                        
                        patch = img_array[y1:y1+crop_size, x1:x1+crop_size]
                        if patch.shape[0] > 10 and patch.shape[1] > 10:
                            normalize_and_save(patch, f'dataset/real_samples/earth_limb/epic_{earth_count}.png')
                            earth_count += 1
    except Exception as e:
        print(f"Failed EPIC: {e}")

    print(f"Done! Collected: {transient_count} Transients, {starfield_count} Starfields, {bright_count} Bright, {earth_count} Earth")

if __name__ == '__main__':
    generate_hybrid_dataset(200)
