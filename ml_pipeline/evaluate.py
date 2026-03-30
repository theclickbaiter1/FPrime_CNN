"""
evaluate.py — Comprehensive model evaluation with metrics and confusion matrix.
"""

import tensorflow as tf
import numpy as np
import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

CLASS_NAMES = ['transient', 'starfield', 'bright_source', 'earth_limb']
NUM_CLASSES = 4


def load_test_data(dataset_dir='dataset'):
    data = np.load(os.path.join(dataset_dir, 'test.npz'))
    return data['x'], data['y']


def evaluate_keras_model(model, x_test, y_test):
    """
    Evaluate the float32 Keras model on the held-out test set.
    Returns raw predictions so the caller can build a confusion matrix.
    """
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"  Float32 Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    return y_pred, y_pred_probs, acc


def evaluate_tflite_model(tflite_path, x_test, y_test):
    """
    Evaluate the INT8 TFLite model on the same held-out test set.
    Runs inference one sample at a time (as on the RP2350) to be realistic.
    Dequantises the INT8 output back to float32 for argmax comparison.
    """
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_scale = input_details[0]['quantization'][0]
    input_zp = input_details[0]['quantization'][1]
    output_scale = output_details[0]['quantization'][0]
    output_zp = output_details[0]['quantization'][1]

    print(f"  Input:  scale={input_scale}, zp={input_zp}, dtype={input_details[0]['dtype']}")
    print(f"  Output: scale={output_scale}, zp={output_zp}")

    y_pred_list = []
    for i in range(len(x_test)):
        sample = x_test[i:i+1]
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
        y_pred_list.append(np.argmax(probs))
        if (i + 1) % 500 == 0:
            print(f"    Processed {i+1}/{len(x_test)}")

    y_pred = np.array(y_pred_list)
    acc = accuracy_score(y_test, y_pred)
    print(f"  INT8 TFLite Accuracy: {acc:.4f} ({acc*100:.1f}%)")
    return y_pred, None, acc


def plot_confusion_matrix(y_true, y_pred, title, save_path):
    """
    Plot a normalised 4x4 confusion matrix with both raw counts and percentages.
    Saved as a PNG for inclusion in reports or the dashboard.
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha='right', fontsize=11)
    ax.set_yticklabels(CLASS_NAMES, fontsize=11)
    ax.set_xlabel('Predicted', fontsize=13, fontweight='bold')
    ax.set_ylabel('True', fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            color = 'white' if cm_norm[i, j] > 0.5 else 'black'
            ax.text(j, i, f'{cm[i,j]}\n({cm_norm[i,j]:.1%})',
                    ha='center', va='center', color=color, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def generate_edge_cases():
    """
    Generate three specialist test sets beyond the main test split:
     faint_transient: transient images with extra additive noise (harder)
     cosmic_ray:      single saturated pixel on a starfield (should → starfield)
     multi_transient: multiple overlapping transient streaks (should → transient)

    These test edge cases use generate_image() from the training pipeline so
    their statistical properties match what the model was trained on.
    """
    edge_cases = []
    # Faint transients — use same generator but with added noise
    for _ in range(20):
        img = generate_image(0)  # Use the actual training pipeline
        # Add extra noise to make it harder
        img = img + np.random.normal(0, 0.03, img.shape).astype(np.float32)
        img = np.clip(img, 0, 1)
        edge_cases.append((img[..., np.newaxis], 0, 'faint_transient'))
    # Cosmic rays (not transient — should be starfield)
    for _ in range(20):
        img = random_background()
        img = add_stars(img, n_stars=40)
        cx, cy = np.random.randint(5, 59), np.random.randint(5, 59)
        img[cy, cx] = 1.0
        img = add_sensor_noise(img)
        img = augment(img)
        edge_cases.append((img.astype(np.float32)[..., np.newaxis], 1, 'cosmic_ray'))
    # Multiple transients — use actual training transient generator
    for _ in range(20):
        img = generate_image(0)  # Use  the actual training pipeline
        edge_cases.append((img[..., np.newaxis], 0, 'multi_transient'))
    return edge_cases


def evaluate_edge_cases(model_or_path, edge_cases, is_tflite=False):
    results = {}
    if is_tflite:
        interpreter = tf.lite.Interpreter(model_path=model_or_path)
        interpreter.allocate_tensors()
        inp = interpreter.get_input_details()
        out = interpreter.get_output_details()
        iscale, izp = inp[0]['quantization']
        oscale, ozp = out[0]['quantization']

    for img, true_label, case_name in edge_cases:
        sample = img[np.newaxis, ...]
        if is_tflite:
            if inp[0]['dtype'] == np.int8:
                sq = np.clip(sample / iscale + izp, -128, 127).astype(np.int8)
            else:
                sq = sample.astype(np.float32)
            interpreter.set_tensor(inp[0]['index'], sq)
            interpreter.invoke()
            o = interpreter.get_tensor(out[0]['index'])
            pred = np.argmax((o.astype(np.float32) - ozp) * oscale if out[0]['dtype'] in [np.int8, np.uint8] else o)
        else:
            pred = np.argmax(model_or_path.predict(sample, verbose=0))
        if case_name not in results:
            results[case_name] = {'correct': 0, 'total': 0}
        results[case_name]['total'] += 1
        if pred == true_label:
            results[case_name]['correct'] += 1

    print("\n  Edge Case Results:")
    for name, r in results.items():
        acc = r['correct'] / r['total']
        s = "PASS" if acc >= 0.7 else "FAIL"
        print(f"    [{s}] {name}: {r['correct']}/{r['total']} ({acc:.0%})")
    return results


def run_full_evaluation(model_path=None, tflite_path=None,
                        dataset_dir='dataset', output_dir='output'):
    os.makedirs(output_dir, exist_ok=True)
    x_test, y_test = load_test_data(dataset_dir)
    print(f"Test set: {x_test.shape}")
    results = {}

    if model_path and os.path.exists(model_path):
        print(f"\n{'='*60}\nFLOAT32 MODEL EVALUATION\n{'='*60}")
        model = tf.keras.models.load_model(model_path)
        y_pred, _, acc = evaluate_keras_model(model, x_test, y_test)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
        print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        plot_confusion_matrix(y_test, y_pred, f'Float32 — {acc:.1%}',
                              os.path.join(output_dir, 'confusion_matrix_fp32.png'))
        results['fp32'] = {'accuracy': float(acc), 'report': report}
        edge_cases = generate_edge_cases()
        ec = evaluate_edge_cases(model, edge_cases, is_tflite=False)
        results['fp32']['edge_cases'] = {k: {'accuracy': v['correct']/v['total']} for k,v in ec.items()}

    if tflite_path and os.path.exists(tflite_path):
        print(f"\n{'='*60}\nINT8 TFLITE MODEL EVALUATION\n{'='*60}")
        sz = os.path.getsize(tflite_path)
        print(f"  Model size: {sz:,} bytes ({sz/1024:.1f} KB)")
        y_pred, _, acc = evaluate_tflite_model(tflite_path, x_test, y_test)
        report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, output_dict=True)
        print("\n" + classification_report(y_test, y_pred, target_names=CLASS_NAMES))
        plot_confusion_matrix(y_test, y_pred, f'INT8 — {acc:.1%}',
                              os.path.join(output_dir, 'confusion_matrix_int8.png'))
        results['int8'] = {'accuracy': float(acc), 'model_size_kb': sz/1024, 'report': report}
        edge_cases = generate_edge_cases()
        ec = evaluate_edge_cases(tflite_path, edge_cases, is_tflite=True)
        results['int8']['edge_cases'] = {k: {'accuracy': v['correct']/v['total']} for k,v in ec.items()}

    if 'fp32' in results and 'int8' in results:
        drop = results['fp32']['accuracy'] - results['int8']['accuracy']
        print(f"\nQuantization drop: {drop:.4f} ({drop*100:.2f}%)")
        print("PASS" if drop < 0.03 else "FAIL — consider more QAT")
        results['quantization_drop'] = float(drop)

    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    return results


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--model', default='output/transient_cnn_fp32.keras')
    p.add_argument('--tflite', default='output/transient_cnn_int8.tflite')
    p.add_argument('--dataset', default='dataset')
    p.add_argument('--output', default='output')
    a = p.parse_args()
    run_full_evaluation(a.model, a.tflite, a.dataset, a.output)
