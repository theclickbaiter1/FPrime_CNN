"""
generate_dataset.py — Synthetic Space Image Dataset Generator

Generates 64x64 grayscale images across 4 classes for training a
transient space event detector CNN:

  Class 0: transient   — asteroid/debris streaks, fast-movers, comets
  Class 1: starfield   — normal static star background (no event)
  Class 2: bright_source — Sun, Moon, bright planet (must ignore)
  Class 3: earth_limb  — Earth horizon / atmospheric glow (must ignore)

Each image simulates realistic sensor characteristics:
  - Poisson shot noise, Gaussian read noise
  - Hot/dead pixels, cosmic ray artifacts
  - Varying exposure levels and star densities
"""

import numpy as np
import os
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Utility helpers — shared image-synthesis primitives used by all class generators
# ---------------------------------------------------------------------------

def add_stars(img, n_stars=None, brightness_range=(0.3, 1.0)):
    """
    Add random point-source stars with a 2D Gaussian PSF.
    Stars are the dominant background signal in all image classes.
    Sigma is varied to simulate slight defocus and seeing conditions.
    """
    h, w = img.shape
    if n_stars is None:
        n_stars = np.random.randint(10, 120)
    for _ in range(n_stars):
        cx = np.random.randint(0, w)
        cy = np.random.randint(0, h)
        brightness = np.random.uniform(*brightness_range)
        sigma = np.random.uniform(0.5, 1.5)
        # small 5x5 PSF stamp
        for dy in range(-2, 3):
            for dx in range(-2, 3):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < h and 0 <= nx < w:
                    val = brightness * np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)
                    img[ny, nx] = min(1.0, img[ny, nx] + val)
    return img


def add_sensor_noise(img, read_noise_std=0.02, hot_pixel_prob=0.0005):
    """
    Add realistic space-camera sensor noise to an image.
    Includes three noise sources:
      - Gaussian read noise: baseline electronic noise floor
      - Poisson shot noise: photon arrival statistics (sqrt of signal)
      - Hot pixels: stuck pixels that always output near-maximum values
    """
    h, w = img.shape
    # Gaussian read noise
    img = img + np.random.normal(0, read_noise_std, (h, w))
    # Poisson-like shot noise (approximated)
    shot_noise = np.random.normal(0, np.sqrt(np.maximum(img, 0) * 0.05 + 1e-6))
    img = img + shot_noise
    # Hot pixels
    hot_mask = np.random.random((h, w)) < hot_pixel_prob
    img[hot_mask] = np.random.uniform(0.8, 1.0, size=hot_mask.sum())
    return np.clip(img, 0.0, 1.0)


def random_background(dark_level=None):
    """
    Generate a dark sky background level with optional spatial gradients.
    Real space cameras pick up faint stray light, zodiacal light, and
    thermal glow — modelled here as a low-level uniform or gradient bias.
    """
    if dark_level is None:
        dark_level = np.random.uniform(0.01, 0.06)
    img = np.full((64, 64), dark_level, dtype=np.float64)
    # subtle gradient
    if np.random.random() < 0.4:
        grad = np.linspace(0, np.random.uniform(0.01, 0.04), 64)
        if np.random.random() < 0.5:
            img += grad[np.newaxis, :]
        else:
            img += grad[:, np.newaxis]
    return img


def augment(img):
    """
    Apply random geometric and photometric augmentations.
    Since a satellite camera can be oriented arbitrarily in three axes,
    flips and 90° rotations are physically realistic augmentations.
    Brightness jitter simulates varying exposure settings.
    """
    if np.random.random() < 0.5:
        img = np.fliplr(img)
    if np.random.random() < 0.5:
        img = np.flipud(img)
    k = np.random.randint(0, 4)
    img = np.rot90(img, k)
    # brightness jitter
    jitter = np.random.uniform(0.85, 1.15)
    img = np.clip(img * jitter, 0.0, 1.0)
    return img


# ---------------------------------------------------------------------------
# Class generators — one function per label class.
# Each generator receives a blank background image and augments it.
#
#   Class 0 — transient:      asteroid / debris streak, fast-mover, anomalous flash
#   Class 1 — starfield:      normal background of point-source stars
#   Class 2 — bright_source:  Sun, Moon, or bright planet — must be IGNORED
#   Class 3 — earth_limb:     Earth horizon/atmosphere — must be IGNORED
# ---------------------------------------------------------------------------

def generate_transient(img):
    """
    Generate a transient event image: moving object streak or bright point anomaly.
    Variants:
      - Linear streak (asteroid / debris)
      - Short bright trail (fast object near FOV)
      - Point-source transient (supernova-like brightening)
    """
    variant = np.random.choice(['streak', 'short_trail', 'point_flash'],
                                p=[0.5, 0.3, 0.2])

    if variant == 'streak':
        # Linear streak across part of the image
        cx, cy = np.random.randint(10, 54), np.random.randint(10, 54)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(8, 35)
        brightness = np.random.uniform(0.4, 1.0)
        width = np.random.uniform(0.8, 2.0)
        for t in np.linspace(-length / 2, length / 2, length * 4):
            px = int(cx + t * np.cos(angle))
            py = int(cy + t * np.sin(angle))
            # Apply width via Gaussian perpendicular spread
            for dw in range(-2, 3):
                nx = int(px + dw * np.sin(angle))
                ny = int(py - dw * np.cos(angle))
                if 0 <= ny < 64 and 0 <= nx < 64:
                    w_val = np.exp(-0.5 * (dw / width) ** 2)
                    # slight brightness variation along streak
                    b_var = brightness * (0.8 + 0.2 * np.random.random())
                    img[ny, nx] = min(1.0, img[ny, nx] + b_var * w_val)

    elif variant == 'short_trail':
        cx, cy = np.random.randint(5, 59), np.random.randint(5, 59)
        angle = np.random.uniform(0, 2 * np.pi)
        length = np.random.randint(3, 10)
        brightness = np.random.uniform(0.5, 1.0)
        for t in np.linspace(0, length, length * 3):
            px = int(cx + t * np.cos(angle))
            py = int(cy + t * np.sin(angle))
            if 0 <= py < 64 and 0 <= px < 64:
                img[py, px] = min(1.0, img[py, px] + brightness)
                # slight spread
                for d in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = py + d[0], px + d[1]
                    if 0 <= ny < 64 and 0 <= nx < 64:
                        img[ny, nx] = min(1.0, img[ny, nx] + brightness * 0.3)

    elif variant == 'point_flash':
        # Bright point source that doesn't look like a normal star
        cx, cy = np.random.randint(5, 59), np.random.randint(5, 59)
        brightness = np.random.uniform(0.6, 1.0)
        sigma = np.random.uniform(1.5, 3.5)  # larger than typical star
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < 64 and 0 <= nx < 64:
                    val = brightness * np.exp(-0.5 * (dx**2 + dy**2) / sigma**2)
                    img[ny, nx] = min(1.0, img[ny, nx] + val)
        # Add asymmetric halo / tail (distinguishes from star)
        tail_angle = np.random.uniform(0, 2 * np.pi)
        for t in range(3, 8):
            nx = int(cx + t * np.cos(tail_angle))
            ny = int(cy + t * np.sin(tail_angle))
            if 0 <= ny < 64 and 0 <= nx < 64:
                img[ny, nx] = min(1.0, img[ny, nx] + brightness * 0.2 * np.exp(-t * 0.3))

    return img


def generate_starfield(img):
    """Generate a normal star field with no transient events."""
    n_stars = np.random.randint(15, 150)
    img = add_stars(img, n_stars=n_stars, brightness_range=(0.15, 0.9))
    return img


def generate_bright_source(img):
    """
    Generate an image dominated by a bright source (Sun, Moon, bright planet).
    These should be large, saturated blobs that the model must learn to IGNORE.
    """
    variant = np.random.choice(['sun', 'moon', 'planet'], p=[0.4, 0.35, 0.25])

    if variant == 'sun':
        # Large, very bright, saturated disc with bloom
        cx = np.random.randint(10, 54)
        cy = np.random.randint(10, 54)
        radius = np.random.uniform(8, 20)
        for y in range(64):
            for x in range(64):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < radius:
                    img[y, x] = 1.0  # fully saturated core
                elif dist < radius * 2.5:
                    # bloom / halo
                    falloff = np.exp(-0.5 * ((dist - radius) / (radius * 0.6)) ** 2)
                    img[y, x] = min(1.0, img[y, x] + 0.8 * falloff)
        # Add diffraction spikes (cross pattern)
        for d in [(1, 0), (0, 1), (1, 1), (-1, 1)]:
            for t in range(int(radius), 55):
                spike_val = 0.6 * np.exp(-t * 0.05)
                nx, ny = int(cx + t * d[0]), int(cy + t * d[1])
                if 0 <= ny < 64 and 0 <= nx < 64:
                    img[ny, nx] = min(1.0, img[ny, nx] + spike_val)
                nx2, ny2 = int(cx - t * d[0]), int(cy - t * d[1])
                if 0 <= ny2 < 64 and 0 <= nx2 < 64:
                    img[ny2, nx2] = min(1.0, img[ny2, nx2] + spike_val)

    elif variant == 'moon':
        # Disc with crescent-like partial illumination
        cx = np.random.randint(15, 49)
        cy = np.random.randint(15, 49)
        radius = np.random.uniform(6, 15)
        phase_offset = np.random.uniform(-radius * 0.5, radius * 0.5)
        for y in range(64):
            for x in range(64):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < radius:
                    # crescent: darken one side
                    illum = 0.5 + 0.5 * np.clip((x - cx - phase_offset) / radius, -1, 1)
                    img[y, x] = min(1.0, img[y, x] + 0.7 * illum)
                elif dist < radius * 1.5:
                    falloff = np.exp(-2 * ((dist - radius) / radius) ** 2)
                    img[y, x] = min(1.0, img[y, x] + 0.2 * falloff)

    elif variant == 'planet':
        # Bright but smaller disc
        cx = np.random.randint(10, 54)
        cy = np.random.randint(10, 54)
        radius = np.random.uniform(3, 7)
        brightness = np.random.uniform(0.7, 1.0)
        for y in range(64):
            for x in range(64):
                dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                if dist < radius:
                    img[y, x] = min(1.0, img[y, x] + brightness)
                elif dist < radius * 2:
                    falloff = np.exp(-2 * ((dist - radius) / radius) ** 2)
                    img[y, x] = min(1.0, img[y, x] + brightness * 0.4 * falloff)

    # May also have some background stars
    if np.random.random() < 0.5:
        img = add_stars(img, n_stars=np.random.randint(5, 30))

    return img


def generate_earth_limb(img):
    """
    Generate an Earth limb / horizon image.
    Bright gradient arc from one edge simulating the Earth's atmosphere.
    """
    # Choose which edge the limb comes from
    edge = np.random.choice(['bottom', 'top', 'left', 'right'])
    limb_extent = np.random.randint(10, 35)  # how far the glow extends
    glow_brightness = np.random.uniform(0.3, 0.9)

    for y in range(64):
        for x in range(64):
            if edge == 'bottom':
                dist = 64 - y
            elif edge == 'top':
                dist = y
            elif edge == 'left':
                dist = x
            elif edge == 'right':
                dist = 64 - x

            if dist < limb_extent:
                # Bright solid earth near edge, transitioning to atmospheric glow
                if dist < limb_extent * 0.3:
                    # "Surface" - bright with slight texture
                    val = glow_brightness * 0.9 + np.random.uniform(0, 0.1)
                    img[y, x] = min(1.0, img[y, x] + val)
                else:
                    # Atmospheric glow falloff
                    t = (dist - limb_extent * 0.3) / (limb_extent * 0.7)
                    val = glow_brightness * (1.0 - t) ** 2
                    img[y, x] = min(1.0, img[y, x] + val)

    # Add a thin bright atmospheric "line" at the transition
    atm_line = int(limb_extent * 0.3)
    if edge in ['bottom', 'top']:
        row = (64 - atm_line) if edge == 'bottom' else atm_line
        if 0 <= row < 64:
            for x in range(64):
                img[row, x] = min(1.0, img[row, x] + glow_brightness * 0.5)
    else:
        col = atm_line if edge == 'left' else (64 - atm_line)
        if 0 <= col < 64:
            for y in range(64):
                img[y, col] = min(1.0, img[y, col] + glow_brightness * 0.5)

    # Stars visible above the limb
    if np.random.random() < 0.7:
        img = add_stars(img, n_stars=np.random.randint(5, 40))

    return img


# ---------------------------------------------------------------------------
# Main generator — orchestrates image creation for all classes
# ---------------------------------------------------------------------------

# canonical class ordering used throughout the whole pipeline;
# the CNN output tensor index matches this list
CLASS_NAMES = ['transient', 'starfield', 'bright_source', 'earth_limb']
CLASS_GENERATORS = [generate_transient, generate_starfield,
                    generate_bright_source, generate_earth_limb]


def generate_image(class_idx):
    """
    Generate a single 64x64 grayscale float32 image for the given class.
    Pipeline: dark background → class content → sensor noise → augmentation.
    Returns a (64, 64) float32 array with values in [0, 1].
    """
    img = random_background()
    img = CLASS_GENERATORS[class_idx](img)
    img = add_sensor_noise(img)
    img = augment(img)
    return img.astype(np.float32)


def generate_dataset(n_per_class_train=5000, n_per_class_test=1000, seed=42):
    """Generate full train/test dataset."""
    np.random.seed(seed)
    n_classes = len(CLASS_NAMES)

    print(f"Generating dataset: {n_per_class_train} train + {n_per_class_test} test per class")
    print(f"Classes: {CLASS_NAMES}")
    print(f"Total: {n_per_class_train * n_classes} train, {n_per_class_test * n_classes} test")

    # Training set
    x_train_list, y_train_list = [], []
    for cls_idx in range(n_classes):
        print(f"  Generating training class {cls_idx} ({CLASS_NAMES[cls_idx]})...")
        for i in range(n_per_class_train):
            img = generate_image(cls_idx)
            x_train_list.append(img)
            y_train_list.append(cls_idx)
            if (i + 1) % 1000 == 0:
                print(f"    {i + 1}/{n_per_class_train}")

    # Test set (different seed region)
    np.random.seed(seed + 999)
    x_test_list, y_test_list = [], []
    for cls_idx in range(n_classes):
        print(f"  Generating test class {cls_idx} ({CLASS_NAMES[cls_idx]})...")
        for i in range(n_per_class_test):
            img = generate_image(cls_idx)
            x_test_list.append(img)
            y_test_list.append(cls_idx)

    # Shuffle
    x_train = np.array(x_train_list)
    y_train = np.array(y_train_list, dtype=np.int32)
    x_test = np.array(x_test_list)
    y_test = np.array(y_test_list, dtype=np.int32)

    perm_train = np.random.permutation(len(x_train))
    x_train, y_train = x_train[perm_train], y_train[perm_train]

    perm_test = np.random.permutation(len(x_test))
    x_test, y_test = x_test[perm_test], y_test[perm_test]

    # Add channel dimension: (N, 64, 64) -> (N, 64, 64, 1)
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    print(f"\nDataset shapes:")
    print(f"  x_train: {x_train.shape}, y_train: {y_train.shape}")
    print(f"  x_test:  {x_test.shape}, y_test:  {y_test.shape}")
    print(f"  Value range: [{x_train.min():.3f}, {x_train.max():.3f}]")

    return (x_train, y_train), (x_test, y_test)


def save_dataset(output_dir='dataset'):
    """Generate and save dataset to .npz files."""
    os.makedirs(output_dir, exist_ok=True)

    (x_train, y_train), (x_test, y_test) = generate_dataset()

    train_path = os.path.join(output_dir, 'train.npz')
    test_path = os.path.join(output_dir, 'test.npz')

    np.savez_compressed(train_path, x=x_train, y=y_train)
    np.savez_compressed(test_path, x=x_test, y=y_test)

    print(f"\nSaved: {train_path} ({os.path.getsize(train_path) / 1e6:.1f} MB)")
    print(f"Saved: {test_path} ({os.path.getsize(test_path) / 1e6:.1f} MB)")

    # Save a few sample images for visual inspection
    save_samples(x_train, y_train, output_dir)


def save_samples(x, y, output_dir, n_per_class=5):
    """Save sample images for each class for visual inspection."""
    try:
        from PIL import Image
    except ImportError:
        print("Pillow not installed, skipping sample image export.")
        return

    samples_dir = os.path.join(output_dir, 'samples')
    os.makedirs(samples_dir, exist_ok=True)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_images = x[y == cls_idx]
        for i in range(min(n_per_class, len(cls_images))):
            img_data = (cls_images[i, :, :, 0] * 255).astype(np.uint8)
            img = Image.fromarray(img_data, mode='L')
            img.save(os.path.join(samples_dir, f'{cls_name}_{i}.png'))

    print(f"Saved sample images to {samples_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate synthetic space event dataset')
    parser.add_argument('--output', default='dataset', help='Output directory')
    parser.add_argument('--train-per-class', type=int, default=5000)
    parser.add_argument('--test-per-class', type=int, default=1000)
    args = parser.parse_args()

    save_dataset(output_dir=args.output)
