import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

def create_ultra_lightweight_cnn(input_shape=(64, 64, 1)):
    """
    Creates an extremely lightweight CNN suitable for MCU deployment via TFLite/CMSIS-NN.
    Uses Depthwise Separable Convolutions to drastically reduce the number of parameters
    and MAC operations.
    """
    model = models.Sequential([
        # Standard convolution to extract initial features and reduce spatial resolution
        # Stride 2 reduces the intermediate tensor size by 4x immediately.
        layers.InputLayer(input_shape=input_shape),
        layers.Conv2D(16, (3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Depthwise Separable Block 1
        layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Depthwise Separable Block 2
        layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        layers.Conv2D(32, (1, 1), strides=(1, 1), padding='same'),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        # Global Average Pooling removes the need for large Dense layers
        layers.GlobalAveragePooling2D(),
        
        # Output layer for Binary Classification
        # We output 2 class probabilities for cross-entropy, or just 1 for sigmoid.
        # Microcontrollers typically prefer fixed size. We will use a dense layer with 2 units & softmax.
        layers.Dense(2, activation='softmax')
    ])
    return model

def load_online_data():
    """
    Loads 'Fashion MNIST' as a high-quality proxy for online data.
    In a production scenario, replace this with a call to:
    tf.keras.utils.get_file('dataset.zip', 'https://example.com/dataset.zip')
    """
    print("Loading online data from tf.keras.datasets...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Filter for binary classification (e.g., class 0 vs others)
    # or just use as is. For transients, we usually want 'event' vs 'no event'.
    # Here we simulate this by taking class 0 (T-shirt) as 'event' and others as 'no event'.
    y_train_binary = (y_train == 0).astype(np.int32)
    y_test_binary = (y_test == 0).astype(np.int32)

    # Resize to 64x64 to match our CNN architecture
    print("Preprocessing data to 64x64 grayscale...")
    x_train = tf.image.resize(x_train[..., np.newaxis], (64, 64)).numpy()
    x_test = tf.image.resize(x_test[..., np.newaxis], (64, 64)).numpy()

    # Normalize to [0, 1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    return (x_train, y_train_binary), (x_test, y_test_binary)

if __name__ == "__main__":
    model = create_ultra_lightweight_cnn()
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    
    # Load real data instead of dummy
    (x_train, y_train), (x_test, y_test) = load_online_data()
    
    # Print summary to verify param count
    model.summary()
    
    print("Starting training...")
    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
    
    # Save the unquantized float32 model
    model.save("transient_cnn_fp32.h5")
    print("Saved transient_cnn_fp32.h5")
