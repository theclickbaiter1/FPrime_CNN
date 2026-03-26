import tensorflow as tf
import numpy as np

# Load original Float32 model
model = tf.keras.models.load_model("transient_cnn_fp32.h5")

def representative_data_gen():
    """
    Generates representative data from Fashion MNIST to calibrate 
    the activation ranges during post-training quantization.
    """
    print("Loading representative data for quantization...")
    (x_train, _), _ = tf.keras.datasets.fashion_mnist.load_data()
    
    # Preprocess exactly as in training
    x_train = tf.image.resize(x_train[..., np.newaxis], (64, 64)).numpy()
    x_train = x_train.astype(np.float32) / 255.0

    # Use 100 samples for calibration
    for i in range(100):
        yield [x_train[i:i+1]]

# Setup TFLite Converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Set optimization flag to reduce size
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Enforce full INT8 quantization
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8  # or tf.uint8 depending on sensor
converter.inference_output_type = tf.int8 # output probabilities as int8

try:
    tflite_quant_model = converter.convert()
    with open('transient_cnn_int8.tflite', 'wb') as f:
        f.write(tflite_quant_model)
    print("Exported fully quantized model to transient_cnn_int8.tflite")
except Exception as e:
    print(f"Quantization failed: {e}")
