"""
train.py — Train an ultra-lightweight CNN for transient space event detection.

Architecture: Depthwise Separable CNN optimized for RP2350 (520KB SRAM, ~100KB model).
Input:  64x64x1 grayscale images
Output: 4 classes [transient, starfield, bright_source, earth_limb]

Trains a high-accuracy float32 model, then relies on post-training INT8
quantization with representative dataset calibration (quantize.py).
"""

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import json


# class label ordering — must match generate_dataset.py
CLASS_NAMES = ['transient', 'starfield', 'bright_source', 'earth_limb']
NUM_CLASSES = 4  # CNN output dimension


def create_ultra_lightweight_cnn(input_shape=(64, 64, 1)):
    """
    Ultra-lightweight CNN using Depthwise Separable Convolutions (DSC).
    DSC factorises a standard conv into depthwise + pointwise passes,
    reducing multiply-accumulate (MAC) ops by ~8-9x vs standard Conv2D.

    Designed for INT8 deployment on the RP2350 (Cortex-M33, 520 KB SRAM).
    All strides double (x2) to aggressively reduce spatial resolution.
    GlobalAveragePooling avoids large Dense layers, keeping memory minimal.

    Parameter budget (approximate):
      Block 1 Conv2D(16):   144 weights +  64 BN = ~208 params
      Block 2 DWSep(32):    144 + 512 + 128 BN   = ~784 params
      Block 3 DWSep(32):    288 + 1024 + 128 BN  = ~1440 params
      Dense(4):             128 + 4               = 132 params
      TOTAL: ~2,756 parameters (~10.77 KB float32, 12 KB INT8)
    """
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),

        # Block 1: Initial feature extraction — stride 2 halves spatial dims to 32x32
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Block 2: Depthwise Separable — stride 2 → 16x16
        layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Block 3: Depthwise Separable — stride 2 → 8x8
        layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.ReLU(),

        # Global Average Pooling → (batch, 32)
        layers.GlobalAveragePooling2D(),

        # Dropout for regularization
        layers.Dropout(0.2),

        # Output: 4 classes
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])
    return model


def load_dataset(dataset_dir='dataset'):
    """
    Load the pre-generated synthetic .npz dataset.
    If the dataset doesn't exist, run generate_dataset.py first.
    """
    train_path = os.path.join(dataset_dir, 'train.npz')
    test_path = os.path.join(dataset_dir, 'test.npz')

    if not os.path.exists(train_path):
        print(f"Dataset not found at {dataset_dir}/. Run generate_dataset.py first!")
        raise FileNotFoundError(f"Missing {train_path}")

    print("Loading synthetic dataset...")
    train_data = np.load(train_path)
    test_data = np.load(test_path)

    x_train, y_train = train_data['x'], train_data['y']
    x_test, y_test = test_data['x'], test_data['y']

    print(f"  Train: {x_train.shape} labels: {y_train.shape}")
    print(f"  Test:  {x_test.shape} labels: {y_test.shape}")
    print(f"  Classes: {CLASS_NAMES}")

    for i, name in enumerate(CLASS_NAMES):
        n_train = np.sum(y_train == i)
        n_test = np.sum(y_test == i)
        print(f"    {name}: {n_train} train, {n_test} test")

    return (x_train, y_train), (x_test, y_test)


def compute_class_weights(y_train):
    """
    Compute per-class weights for the loss function.
    Handles any class imbalance that could bias the model toward
    majority classes. With a balanced synthetic dataset these will all be 1.0.
    """
    from sklearn.utils.class_weight import compute_class_weight
    weights = compute_class_weight('balanced', classes=np.arange(NUM_CLASSES), y=y_train)
    weight_dict = {i: w for i, w in enumerate(weights)}
    print(f"  Class weights: {weight_dict}")
    return weight_dict


def cosine_decay_schedule(epoch, lr):
    """
    Cosine annealing learning rate schedule.
    Starts at max_lr=1e-3, smoothly decays to min_lr=1e-6 over 30 epochs.
    This warmly converges: large steps early, fine-tuning steps later.
    """
    max_epochs = 30
    min_lr = 1e-6
    max_lr = 1e-3
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + np.cos(np.pi * epoch / max_epochs))


def train(dataset_dir='dataset', output_dir='output', epochs=30, batch_size=32):
    """
    Full training pipeline.
    Trains for up to `epochs` epochs, saves the best checkpoint by val_accuracy.
    Uses EarlyStopping with patience=8 to prevent overfitting.
    Outputs both .h5 and .keras model formats for maximum compatibility.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    (x_train, y_train), (x_test, y_test) = load_dataset(dataset_dir)

    # Build model
    print("\nBuilding CNN architecture...")
    model = create_ultra_lightweight_cnn()
    model.summary()

    class_weights = compute_class_weights(y_train)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    # Callbacks
    cb = [
        callbacks.EarlyStopping(
            monitor='val_accuracy', patience=8,
            restore_best_weights=True, verbose=1
        ),
        callbacks.LearningRateScheduler(cosine_decay_schedule, verbose=0),
        callbacks.ModelCheckpoint(
            os.path.join(output_dir, 'best_model.keras'),
            monitor='val_accuracy', save_best_only=True, verbose=1
        ),
    ]

    print(f"\n--- Training for {epochs} epochs ---")
    history = model.fit(
        x_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, y_test),
        class_weight=class_weights,
        callbacks=cb,
        verbose=1
    )

    # Final evaluation
    final_loss, final_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"\n{'='*60}")
    print(f"  FINAL VALIDATION ACCURACY: {final_acc:.4f} ({final_acc*100:.1f}%)")
    print(f"  FINAL VALIDATION LOSS:     {final_loss:.4f}")
    print(f"{'='*60}")

    # Save model in both formats
    h5_path = os.path.join(output_dir, 'transient_cnn_fp32.h5')
    model.save(h5_path)
    print(f"\nSaved H5 model: {h5_path}")

    keras_path = os.path.join(output_dir, 'transient_cnn_fp32.keras')
    model.save(keras_path)
    print(f"Saved Keras model: {keras_path}")

    # Save training history
    history_data = {
        'accuracy': [float(v) for v in history.history.get('accuracy', [])],
        'val_accuracy': [float(v) for v in history.history.get('val_accuracy', [])],
        'loss': [float(v) for v in history.history.get('loss', [])],
        'val_loss': [float(v) for v in history.history.get('val_loss', [])],
        'final_accuracy': float(final_acc),
        'final_loss': float(final_loss),
        'class_names': CLASS_NAMES,
        'total_params': int(model.count_params()),
    }
    hist_path = os.path.join(output_dir, 'training_history.json')
    with open(hist_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    print(f"Saved training history: {hist_path}")

    return model, history


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Train transient event detector CNN')
    parser.add_argument('--dataset', default='dataset', help='Dataset directory')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    train(dataset_dir=args.dataset, output_dir=args.output,
          epochs=args.epochs, batch_size=args.batch_size)
