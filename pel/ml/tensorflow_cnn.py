"""
Implementation of a ResNet-style Convolutional Neural Network using TensorFlow.
This file contains:
- A ResNet block implementation
- A complete ResNet model implementation
- Training code using the CIFAR-10 dataset

The model implements a simplified ResNet architecture with:
- Initial convolution layer
- 3 ResNet blocks
- Global average pooling
- Dense classification layer
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split


def create_resnet_block(inputs, filters, kernel_size=3, strides=1) -> tf.Tensor:
    """Creates a residual block with two convolution layers."""
    # Store the input for the skip connection
    shortcut = inputs
    
    # First convolution layer
    x = layers.Conv2D(filters, kernel_size, strides=strides, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Second convolution layer
    x = layers.Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    
    # Projection shortcut if dimensions change
    if strides != 1 or inputs.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=strides, padding='same')(inputs)
        shortcut = layers.BatchNormalization()(shortcut)
    
    # Add skip connection
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def create_resnet_model(input_shape=(32, 32, 3), num_classes=10) -> tf.keras.Model:
    """Creates a simplified ResNet model."""
    inputs = layers.Input(shape=input_shape)
    
    # Initial convolution block
    x = layers.Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2, padding='same')(x)
    
    # ResNet blocks
    x = create_resnet_block(x, 32)
    x = create_resnet_block(x, 64, strides=2)
    x = create_resnet_block(x, 128, strides=2)
    
    # Global average pooling and final dense layer
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs, outputs)


if __name__ == "__main__":
    # Load and preprocess CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Normalize pixel values
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Split the training data into training and validation sets
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    
    # Convert labels to categorical
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    y_val = tf.keras.utils.to_categorical(y_val, 10)
    
    # Create and compile model
    model = create_resnet_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3
        )
    ]
    
    # In the main block, before model.fit:
    data_augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ])

    batch_size = 128

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    train_dataset = train_dataset.batch(batch_size)

    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    val_dataset = val_dataset.batch(batch_size)
    
    # Update model.fit to use data augmentation
    model.fit(
        train_dataset,
        epochs=50,
        validation_data=val_dataset,
        callbacks=callbacks
    )
    
    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(x_test, y_test)
    print(f"\nTest accuracy: {test_accuracy:.4f}")

