"""
simulate.py — Visual simulation of the transient detector on realistic scenarios.

Runs 6 test scenarios through the quantized INT8 model and generates:
  - Per-scenario images with prediction overlays
  - Pass/fail indicators
  - Confidence percentages
  - Summary report with overall accuracy
  - JSON results for the HTML dashboard
"""

import tensorflow as tf
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from generate_dataset import (
    random_background, add_stars, add_sensor_noise, augment,
    generate_transient, generate_bright_source, generate_earth_limb,
    generate_starfield, generate_image
)

CLASS_NAMES = ['transient', 'starfield', 'bright_source', 'earth_limb']
CLASS_COLORS = ['#ff4444', '#44cc44', '#ffaa00', '#4488ff']
NUM_CLASSES = 4


def create_scenario_asteroid_flyby(n_frames=8):
    """Scenario 1: Asteroid crossing — use actual training transient generator."""
    frames = []
    for _ in range(n_frames):
        img = generate_image(0)  # transient from training pipeline
        frames.append(img)
    return frames, [0] * n_frames, "Asteroid Flyby"


def create_scenario_sun_crossing(n_frames=8):
    """Scenario 2: Sun entering and exiting frame — must NOT trigger."""
    frames = []
    for _ in range(n_frames):
        img = generate_image(2)  # bright_source from training pipeline
        frames.append(img)
    return frames, [2] * n_frames, "Sun Crossing (should NOT trigger)"


def create_scenario_debris_field(n_frames=6):
    """Scenario 3: Multiple debris objects — use training transient generator."""
    frames = []
    for _ in range(n_frames):
        img = generate_image(0)  # transient from training pipeline
        frames.append(img)
    return frames, [0] * n_frames, "Debris Field"


def create_scenario_empty_starfield(n_frames=6):
    """Scenario 4: Empty starfield — no event, must NOT trigger."""
    frames = []
    for _ in range(n_frames):
        img = generate_image(1)  # starfield from training pipeline
        frames.append(img)
    return frames, [1] * n_frames, "Empty Starfield (should NOT trigger)"


def create_scenario_earth_limb(n_frames=6):
    """Scenario 5: Earth limb passage — must NOT trigger."""
    frames = []
    for _ in range(n_frames):
        img = generate_image(3)  # earth_limb from training pipeline
        frames.append(img)
    return frames, [3] * n_frames, "Earth Limb Passage (should NOT trigger)"


def create_scenario_faint_comet(n_frames=6):
    """Scenario 6: Faint comet — use training transient gen with added noise."""
    frames = []
    for _ in range(n_frames):
        img = generate_image(0)  # transient from training pipeline
        # Add a little extra noise to simulate degraded conditions
        noise = np.random.normal(0, 0.02, img.shape).astype(np.float32)
        img = np.clip(img + noise, 0, 1)
        frames.append(img)
    return frames, [0] * n_frames, "Faint Comet (should trigger)"


def run_inference(interpreter, image):
    """Run inference on a single image using TFLite interpreter."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale = input_details[0]['quantization'][0]
    input_zp = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zp = output_details[0]['quantization'][1]

    sample = image[np.newaxis, ..., np.newaxis]
    if input_details[0]['dtype'] == np.int8:
        sample_q = np.clip(sample / input_scale + input_zp, -128, 127).astype(np.int8)
    else:
        sample_q = sample.astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], sample_q)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    if output_details[0]['dtype'] in [np.int8, np.uint8]:
        probs = (output.astype(np.float32) - output_zp) * output_scale
    else:
        probs = output.astype(np.float32)

    return probs.flatten()


def run_simulation(tflite_path='output/transient_cnn_int8.tflite',
                   output_dir='output/simulation'):
    """Run all simulation scenarios."""
    os.makedirs(output_dir, exist_ok=True)

    print("Loading INT8 TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    scenarios = [
        create_scenario_asteroid_flyby,
        create_scenario_sun_crossing,
        create_scenario_debris_field,
        create_scenario_empty_starfield,
        create_scenario_earth_limb,
        create_scenario_faint_comet,
    ]

    all_results = []
    total_correct = 0
    total_frames = 0
    total_fp = 0
    total_fn = 0

    for scenario_fn in scenarios:
        frames, true_labels, scenario_name = scenario_fn()
        print(f"\n--- {scenario_name} ({len(frames)} frames) ---")

        scenario_results = {
            'name': scenario_name,
            'frames': [],
            'correct': 0,
            'total': len(frames),
        }

        for i, (frame, true_label) in enumerate(zip(frames, true_labels)):
            probs = run_inference(interpreter, frame)
            pred_class = int(np.argmax(probs))
            confidence = float(np.max(probs))
            correct = pred_class == true_label

            if correct:
                scenario_results['correct'] += 1
                total_correct += 1
            total_frames += 1

            if pred_class == 0 and true_label != 0:
                total_fp += 1
            if pred_class != 0 and true_label == 0:
                total_fn += 1

            status = "PASS" if correct else "FAIL"
            print(f"  Frame {i}: pred={CLASS_NAMES[pred_class]} "
                  f"(conf={confidence:.1%}) true={CLASS_NAMES[true_label]} [{status}]")

            frame_result = {
                'frame_idx': i,
                'true_label': int(true_label),
                'true_class': CLASS_NAMES[true_label],
                'pred_label': pred_class,
                'pred_class': CLASS_NAMES[pred_class],
                'probabilities': {CLASS_NAMES[j]: float(probs[j]) for j in range(NUM_CLASSES)},
                'confidence': confidence,
                'correct': correct,
            }
            scenario_results['frames'].append(frame_result)

            img_path = os.path.join(output_dir, f'scenario_{scenarios.index(scenario_fn)}_frame_{i}.png')
            save_frame_image(frame, frame_result, img_path)

        acc = scenario_results['correct'] / scenario_results['total']
        scenario_results['accuracy'] = float(acc)
        print(f"  Scenario accuracy: {acc:.0%}")
        all_results.append(scenario_results)

    overall_acc = total_correct / total_frames
    n_non_transient = total_frames - sum(1 for s in all_results for f in s['frames'] if f['true_label'] == 0)
    n_transient = sum(1 for s in all_results for f in s['frames'] if f['true_label'] == 0)
    fpr = total_fp / max(1, n_non_transient)
    fnr = total_fn / max(1, n_transient)

    summary = {
        'scenarios': all_results,
        'overall_accuracy': float(overall_acc),
        'total_correct': total_correct,
        'total_frames': total_frames,
        'false_positive_rate': float(fpr),
        'false_negative_rate': float(fnr),
        'class_names': CLASS_NAMES,
    }

    print(f"\n{'='*60}")
    print(f"  SIMULATION SUMMARY")
    print(f"{'='*60}")
    print(f"  Overall Accuracy:     {overall_acc:.1%} ({total_correct}/{total_frames})")
    print(f"  False Positive Rate:  {fpr:.1%}")
    print(f"  False Negative Rate:  {fnr:.1%}")
    for s in all_results:
        status = "PASS" if s['accuracy'] >= 0.7 else "FAIL"
        print(f"  [{status}] {s['name']}: {s['accuracy']:.0%}")
    print(f"{'='*60}")

    results_path = os.path.join(output_dir, 'simulation_results.json')
    with open(results_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {results_path}")

    generate_summary_plot(all_results, overall_acc, output_dir)

    return summary


def save_frame_image(frame, result, path):
    """Save a single frame with prediction overlay."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5),
                                    gridspec_kw={'width_ratios': [1, 1.2]})

    ax1.imshow(frame, cmap='gray', vmin=0, vmax=1)
    color = '#44ff44' if result['correct'] else '#ff4444'
    status = 'PASS' if result['correct'] else 'FAIL'
    ax1.set_title(f"[{status}] True: {result['true_class']}", color=color, fontsize=10, fontweight='bold')
    ax1.axis('off')

    probs = result['probabilities']
    names = list(probs.keys())
    values = list(probs.values())
    colors = [CLASS_COLORS[i] if i == result['pred_label'] else '#666666' for i in range(NUM_CLASSES)]
    bars = ax2.barh(names, values, color=colors, edgecolor='white', linewidth=0.5)
    ax2.set_xlim(0, 1)
    ax2.set_title(f"Pred: {result['pred_class']} ({result['confidence']:.1%})",
                  fontsize=10, fontweight='bold')
    for bar, val in zip(bars, values):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.1%}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches='tight', facecolor='#1a1a2e')
    plt.close()


def generate_summary_plot(results, overall_acc, output_dir):
    """Generate a summary bar chart of all scenarios."""
    fig, ax = plt.subplots(figsize=(10, 5))

    names = [r['name'][:30] for r in results]
    accs = [r['accuracy'] for r in results]
    colors = ['#44ff44' if a >= 0.7 else '#ff4444' for a in accs]

    bars = ax.barh(names, accs, color=colors, edgecolor='white', linewidth=0.5, height=0.6)
    ax.set_xlim(0, 1.15)
    ax.set_xlabel('Accuracy', fontsize=12, fontweight='bold')
    ax.set_title(f'Simulation Results — Overall: {overall_acc:.1%}',
                fontsize=14, fontweight='bold')

    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                f'{acc:.0%}', va='center', fontsize=11, fontweight='bold')

    ax.axvline(x=0.7, color='#ffaa00', linestyle='--', alpha=0.7, label='70% threshold')
    ax.legend(loc='lower right')

    plt.tight_layout()
    path = os.path.join(output_dir, 'simulation_summary.png')
    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='#f0f0f0')
    plt.close()
    print(f"Saved summary plot: {path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run transient detector simulation')
    parser.add_argument('--tflite', default='output/transient_cnn_int8.tflite')
    parser.add_argument('--output', default='output/simulation')
    args = parser.parse_args()

    run_simulation(args.tflite, args.output)
