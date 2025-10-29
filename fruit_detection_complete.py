#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CS4287 Neural Computing - Assignment 2: Fruit Detection using CNNs
Team Members: [INSERT NAMES AND ID NUMBERS]

This is a complete implementation for fruit detection using a Kaggle dataset.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, losses, metrics
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print(f"TensorFlow version: {tf.__version__}")
print(f"Keras version: {keras.__version__}")

# ============================================================================
# DATA LOADING FOR FRUIT DETECTION
# ============================================================================

def load_fruit_dataset(dataset_path, image_size=(224, 224), batch_size=32):
    """
    Load fruit dataset from Kaggle using Keras utilities.
    
    Expected folder structure:
    dataset_path/
        train/
            apple/
                img1.jpg, img2.jpg, ...
            banana/
                img1.jpg, img2.jpg, ...
            orange/
                ...
        test/
            apple/
            banana/
            orange/
    """
    # Load training dataset
    train_dataset = keras.utils.image_dataset_from_directory(
        f'{dataset_path}/train',
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )
    
    # Load test dataset
    test_dataset = keras.utils.image_dataset_from_directory(
        f'{dataset_path}/test',
        labels='inferred',
        label_mode='categorical',
        image_size=image_size,
        batch_size=batch_size,
        shuffle=False,
        seed=42
    )
    
    # Get class names
    class_names = train_dataset.class_names
    num_classes = len(class_names)
    
    print(f"\nDataset loaded successfully!")
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {class_names}")
    print(f"Number of training batches: {tf.data.experimental.cardinality(train_dataset).numpy()}")
    print(f"Number of test batches: {tf.data.experimental.cardinality(test_dataset).numpy()}")
    
    return train_dataset, test_dataset, class_names, num_classes

def visualize_fruit_samples(train_dataset, class_names, num_samples=16):
    """Visualize sample fruit images from each class."""
    plt.figure(figsize=(15, 15))
    
    # Get first batch
    for images, labels in train_dataset.take(1):
        for i in range(min(num_samples, len(images))):
            plt.subplot(4, 4, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            label = np.argmax(labels[i].numpy())
            plt.title(f"{class_names[label]}")
            plt.axis('off')
    
    plt.suptitle('Sample Fruits from Training Set', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_class_distribution(train_dataset, class_names):
    """Plot the distribution of fruit classes in the training set."""
    class_counts = {}
    total = 0
    
    for images, labels in train_dataset:
        for label in labels:
            idx = np.argmax(label)
            class_name = class_names[idx]
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            total += 1
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(class_counts.keys(), class_counts.values(), color='steelblue', edgecolor='black')
    plt.title('Distribution of Fruit Classes in Training Set', fontsize=14, fontweight='bold')
    plt.xlabel('Fruit Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    # Add count labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    return class_counts

# ============================================================================
# DATA AUGMENTATION FOR FRUIT IMAGES
# ============================================================================

def create_data_augmentation():
    """
    Create data augmentation for fruit images to prevent overfitting.
    Augmentations: rotation, shifts, flips, zoom, brightness adjustments.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,                    # Normalize pixel values to [0,1]
        rotation_range=30,                 # Random rotation up to 30 degrees
        width_shift_range=0.2,             # Horizontal shift
        height_shift_range=0.2,            # Vertical shift
        shear_range=0.2,                    # Shear transformation
        zoom_range=0.2,                     # Zoom in/out
        horizontal_flip=True,               # Random horizontal flip
        vertical_flip=False,                # Don't flip vertically for fruits
        fill_mode='nearest'                 # Fill pixels outside boundaries
    )
    
    # For validation/test - only rescaling
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    return train_datagen, val_datagen

# ============================================================================
# BUILD CNN ARCHITECTURE FOR FRUIT DETECTION
# ============================================================================

def build_fruit_cnn(input_shape, num_classes):
    """
    Build a CNN model for fruit detection and classification.
    
    Architecture:
    - 3 convolutional blocks for feature extraction (edges → textures → patterns)
    - Batch normalization for stable training
    - Max pooling for dimensionality reduction
    - Dropout for regularization
    - Dense layers for classification
    
    Args:
        input_shape: Tuple (height, width, channels)
        num_classes: Number of fruit classes
    
    Returns:
        Compiled Keras model
    """
    model = models.Sequential([
        # First Convolutional Block - extracts basic features (edges, lines, colors)
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape,
                     kernel_initializer='he_normal', name='conv1_1'),
        layers.BatchNormalization(name='bn1_1'),
        layers.Conv2D(32, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv1_2'),
        layers.BatchNormalization(name='bn1_2'),
        layers.MaxPooling2D((2, 2), name='pool1'),
        layers.Dropout(0.25, name='dropout1'),
        
        # Second Convolutional Block - extracts textures and patterns
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv2_1'),
        layers.BatchNormalization(name='bn2_1'),
        layers.Conv2D(64, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv2_2'),
        layers.BatchNormalization(name='bn2_2'),
        layers.MaxPooling2D((2, 2), name='pool2'),
        layers.Dropout(0.25, name='dropout2'),
        
        # Third Convolutional Block - extracts fruit-specific features (shape, color patterns)
        layers.Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv3_1'),
        layers.BatchNormalization(name='bn3_1'),
        layers.Conv2D(128, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv3_2'),
        layers.BatchNormalization(name='bn3_2'),
        layers.MaxPooling2D((2, 2), name='pool3'),
        layers.Dropout(0.25, name='dropout3'),
        
        # Fourth Convolutional Block - high-level features
        layers.Conv2D(256, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv4_1'),
        layers.BatchNormalization(name='bn4_1'),
        layers.Conv2D(256, (3, 3), activation='relu',
                     kernel_initializer='he_normal', name='conv4_2'),
        layers.BatchNormalization(name='bn4_2'),
        layers.MaxPooling2D((2, 2), name='pool4'),
        layers.Dropout(0.25, name='dropout4'),
        
        # Fully Connected Layers - classification
        layers.Flatten(name='flatten'),
        layers.Dense(512, activation='relu',
                    kernel_initializer='he_normal', name='fc1'),
        layers.BatchNormalization(name='bn_fc1'),
        layers.Dropout(0.5, name='dropout_fc1'),
        layers.Dense(256, activation='relu',
                    kernel_initializer='he_normal', name='fc2'),
        layers.BatchNormalization(name='bn_fc2'),
        layers.Dropout(0.5, name='dropout_fc2'),
        
        # Output Layer - probability distribution over fruit classes
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    
    return model

# ============================================================================
# TRAINING
# ============================================================================

def compile_model(model, learning_rate=0.001):
    """
    Compile the model with optimizer and loss function.
    
    Adam optimizer: adaptive learning rate for each parameter
    Categorical cross-entropy: standard for multi-class classification
    """
    optimizer = optimizers.Adam(
        learning_rate=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy', keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')]
    )
    
    return model

def create_callbacks(model_save_path='best_fruit_model.h5'):
    """Create callbacks for training."""
    return [
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=0.00001,
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

def train_model(model, train_dataset, val_dataset, epochs=100, batch_size=32):
    """Train the fruit detection model."""
    callbacks = create_callbacks()
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# ============================================================================
# EVALUATION AND VISUALIZATION
# ============================================================================

def plot_training_history(history):
    """Plot training and validation accuracy/loss."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Accuracy plot
    axes[0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
    axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0].set_title('Model Accuracy for Fruit Detection', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Accuracy', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[1].plot(history.history['loss'], label='Train Loss', linewidth=2)
    axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[1].set_title('Model Loss for Fruit Detection', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def generate_confusion_matrix(model, test_dataset, class_names):
    """Generate and visualize confusion matrix."""
    # Get all predictions and true labels
    y_true = []
    y_pred = []
    
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(preds, axis=1))
        y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Number of Predictions'})
    plt.title('Confusion Matrix for Fruit Detection', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    return cm, y_true, y_pred

def visualize_predictions(model, test_dataset, class_names, num_samples=16):
    """Visualize predictions on random test samples."""
    fig, axes = plt.subplots(4, 4, figsize=(15, 15))
    
    # Get samples from test set
    for images, labels in test_dataset.take(1):
        for i in range(min(num_samples, len(images))):
            ax = axes[i // 4, i % 4]
            
            # Display image
            image = images[i].numpy().astype("uint8")
            ax.imshow(image)
            
            # Get prediction
            pred = model.predict(np.expand_dims(images[i], axis=0), verbose=0)
            pred_class_idx = np.argmax(pred[0])
            pred_class = class_names[pred_class_idx]
            confidence = pred[0][pred_class_idx]
            
            # Get true label
            true_label_idx = np.argmax(labels[i].numpy())
            true_label = class_names[true_label_idx]
            
            # Color title based on correctness
            color = 'green' if pred_class == true_label else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_class}\nConf: {confidence:.2f}',
                        color=color, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle('Fruit Detection Predictions (Green=Correct, Red=Incorrect)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ============================================================================
# HYPERPARAMETER ANALYSIS
# ============================================================================

def analyze_hyperparameters(dataset_path, learning_rates=[0.01, 0.001, 0.0001]):
    """Analyze impact of different learning rates."""
    results = []
    
    for lr in learning_rates:
        print(f"\n{'='*50}")
        print(f"Testing with learning rate: {lr}")
        print(f"{'='*50}")
        
        # Load dataset
        train_dataset, test_dataset, class_names, num_classes = load_fruit_dataset(dataset_path)
        
        # Build model
        model = build_fruit_cnn((224, 224, 3), num_classes)
        model = compile_model(model, learning_rate=lr)
        
        # Train for fewer epochs for analysis
        history = train_model(model, train_dataset, test_dataset, epochs=10)
        
        # Evaluate
        test_loss, test_acc, _ = model.evaluate(test_dataset, verbose=0)
        results.append({
            'lr': lr,
            'test_acc': test_acc,
            'test_loss': test_loss
        })
    
    # Visualize results
    plt.figure(figsize=(10, 6))
    plt.plot([r['lr'] for r in results], [r['test_acc'] for r in results], marker='o', linewidth=2)
    plt.xlabel('Learning Rate', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    plt.title('Impact of Learning Rate on Fruit Detection Performance', fontweight='bold')
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "data/fruits_classification"  # Your Kaggle fruit dataset
    IMAGE_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 100
    
    print("="*60)
    print("Fruit Detection using Convolutional Neural Networks")
    print("CS4287 Neural Computing - Assignment 2")
    print("="*60)
    
    # Load dataset
    print("\n1. Loading fruit dataset...")
    train_dataset, test_dataset, class_names, num_classes = load_fruit_dataset(
        DATASET_PATH, IMAGE_SIZE, BATCH_SIZE
    )
    
    # Visualize data
    print("\n2. Visualizing fruit samples...")
    visualize_fruit_samples(train_dataset, class_names)
    plot_class_distribution(train_dataset, class_names)
    
    # Build model
    print("\n3. Building CNN architecture...")
    model = build_fruit_cnn((*IMAGE_SIZE, 3), num_classes)
    model.summary()
    
    # Compile model
    print("\n4. Compiling model...")
    model = compile_model(model)
    
    # Train model
    print("\n5. Training model...")
    print("This may take a while depending on your dataset size and hardware.")
    history = train_model(model, train_dataset, test_dataset, EPOCHS, BATCH_SIZE)
    
    # Visualize training
    print("\n6. Plotting training history...")
    plot_training_history(history)
    
    # Evaluate model
    print("\n7. Evaluating model...")
    cm, y_true, y_pred = generate_confusion_matrix(model, test_dataset, class_names)
    
    # Visualize predictions
    print("\n8. Visualizing predictions...")
    visualize_predictions(model, test_dataset, class_names)
    
    print("\n" + "="*60)
    print("Training and evaluation completed successfully!")
    print("="*60)
