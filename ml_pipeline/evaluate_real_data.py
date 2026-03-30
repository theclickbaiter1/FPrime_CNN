import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Match the output mapping
CLASS_NAMES = ['transient', 'starfield', 'bright_source', 'earth_limb']

def load_real_data_dir(base_dir='dataset/real_samples'):
    X = []
    y = []
    
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(base_dir, class_name)
        if not os.path.exists(class_dir):
            continue
            
        for fname in os.listdir(class_dir):
            if not fname.endswith('.png'): continue
            path = os.path.join(class_dir, fname)
            
            # Load and normalize
            img = Image.open(path).convert('L') # ensure grayscale
            img_arr = np.array(img, dtype=np.float32) / 255.0
            
            X.append(img_arr)
            y.append(class_idx)
            
    X = np.expand_dims(np.array(X), axis=-1)
    y = np.array(y)
    return X, y

def plot_confusion_matrix(y_true, y_pred, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    # Check if we have all 4 classes in test
    classes_present = np.unique(np.concatenate([y_true, y_pred]))
    labels = [CLASS_NAMES[i] for i in classes_present]
    
    # Normalise
    cm_norm = np.zeros_like(cm, dtype=float)
    for i in range(len(cm)):
        if cm[i].sum() > 0:
            cm_norm[i] = cm[i].astype(float) / cm[i].sum()
            
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
    fig.colorbar(cax)
    
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='left')
    ax.set_yticklabels(labels)
    
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            pct = cm_norm[i, j] * 100
            if val > 0:
                ax.text(j, i, f"{val}\n({pct:.1f}%)", ha="center", va="center",
                        color="white" if pct > 50 else "black")
                        
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title, pad=20)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    print("Loading test dataset from real_samples...")
    X_test, y_test = load_real_data_dir()
    
    if len(X_test) == 0:
        print("No real test data found.")
        return
        
    print(f"Loaded {len(X_test)} total real-life test images.")
    # Show counts
    counts = np.bincount(y_test, minlength=4)
    for i, c in enumerate(counts):
        print(f"  {CLASS_NAMES[i]}: {c} images")
        
    model_path = 'output/transient_cnn_fp32.keras'
    print(f"Loading float32 baseline model: {model_path}")
    model = tf.keras.models.load_model(model_path)
    
    print("Running inference...")
    y_pred_probs = model.predict(X_test, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\n--- Baseline Model on Real Data ---")
    print(f"Overall Accuracy: {acc*100:.2f}%")
    print("\nClassification Report:")
    
    # use labels parameter to prevent errors if a class is missing in test data
    classes_present = np.unique(y_test)
    target_names = [CLASS_NAMES[i] for i in classes_present]
    report = classification_report(y_test, y_pred, labels=classes_present, target_names=target_names)
    print(report)
    
    os.makedirs('output/real_data_eval', exist_ok=True)
    plot_path = 'output/real_data_eval/confusion_matrix_baseline.png'
    plot_confusion_matrix(y_test, y_pred, "Baseline Synthetic Model on Real Data", plot_path)
    print(f"Saved confusion matrix to {plot_path}")

if __name__ == '__main__':
    main()
