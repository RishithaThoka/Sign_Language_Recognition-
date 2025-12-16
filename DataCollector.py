import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import cv2
import os
import json
from collections import deque
from datetime import datetime
import pickle

# ===========================
# 1. DATA COLLECTION SYSTEM
# ===========================

class SignLanguageDataCollector:
    """Interactive tool to collect sign language training data"""
    
    def __init__(self, data_dir="sign_data", sequence_length=30):
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.current_sign = None
        self.sequences = []
        self.recording = False
        self.countdown = 0
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Hand detection setup
        self.lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        self.upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        
        # Feature extractor
        self._setup_feature_extractor()
        
        # Stats
        self.stats_file = os.path.join(data_dir, "collection_stats.json")
        self.load_stats()
    
    def _setup_feature_extractor(self):
        """Setup CNN feature extractor"""
        base_model = keras.applications.MobileNetV2(
            input_shape=(224, 224, 3),
            include_top=False,
            weights='imagenet'
        )
        base_model.trainable = False
        
        self.feature_model = keras.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(128, activation='relu'),
            layers.Dense(81)
        ])
    
    def load_stats(self):
        """Load collection statistics"""
        if os.path.exists(self.stats_file):
            with open(self.stats_file, 'r') as f:
                self.stats = json.load(f)
        else:
            self.stats = {}
    
    def save_stats(self):
        """Save collection statistics"""
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=2)
    
    def detect_hand(self, frame):
        """Enhanced hand detection"""
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        mask = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Blur for smoother contours
        mask = cv2.GaussianBlur(mask, (5, 5), 0)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Get largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            # Filter small contours (noise)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(largest)
                
                # Add padding
                padding = 20
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(frame.shape[1] - x, w + 2*padding)
                h = min(frame.shape[0] - y, h + 2*padding)
                
                hand_roi = frame[y:y+h, x:x+w]
                return hand_roi, (x, y, w, h), mask
        
        return None, None, mask
    
    def extract_features(self, frame):
        """Extract features from frame"""
        hand_roi, bbox, mask = self.detect_hand(frame)
        
        if hand_roi is not None and hand_roi.size > 0:
            try:
                # Resize and normalize
                hand_resized = cv2.resize(hand_roi, (224, 224))
                hand_resized = hand_resized.astype(np.float32) / 255.0
                hand_resized = np.expand_dims(hand_resized, axis=0)
                
                # Extract features
                features = self.feature_model.predict(hand_resized, verbose=0)[0]
                return features, bbox, mask
            except:
                pass
        
        return np.zeros(81, dtype=np.float32), None, mask
    
    def collect_sign(self, sign_name, num_samples=50):
        """Collect samples for a specific sign"""
        self.current_sign = sign_name
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        print(f"\n{'='*60}")
        print(f"COLLECTING DATA FOR: {sign_name.upper()}")
        print(f"{'='*60}")
        print(f"Target: {num_samples} samples")
        print(f"Current: {self.stats.get(sign_name, 0)} samples collected")
        print(f"\nInstructions:")
        print("  - Position your hand in the green box")
        print("  - Press SPACE to start recording")
        print("  - Hold the sign for 1 second")
        print("  - Repeat {num_samples} times with variations")
        print("  - Press 'q' to finish and save")
        print("  - Press 'r' to redo last sample")
        print(f"{'='*60}\n")
        
        collected = []
        sample_count = 0
        frame_buffer = deque(maxlen=self.sequence_length)
        
        try:
            while sample_count < num_samples:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                display = frame.copy()
                
                # Extract features
                features, bbox, mask = self.extract_features(frame)
                
                # Draw hand detection
                if bbox:
                    x, y, w, h = bbox
                    color = (0, 255, 0) if not self.recording else (0, 0, 255)
                    cv2.rectangle(display, (x, y), (x+w, y+h), color, 2)
                    cv2.putText(display, "Hand Detected", (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Recording logic
                if self.recording:
                    frame_buffer.append(features)
                    self.countdown -= 1
                    
                    # Show countdown
                    frames_left = max(0, self.sequence_length - len(frame_buffer))
                    cv2.putText(display, f"RECORDING: {frames_left} frames left", 
                               (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    
                    # Finished recording one sample
                    if len(frame_buffer) >= self.sequence_length:
                        sequence = np.array(list(frame_buffer))
                        collected.append(sequence)
                        sample_count += 1
                        self.recording = False
                        frame_buffer.clear()
                        print(f"✓ Sample {sample_count}/{num_samples} collected!")
                
                # Display info
                cv2.rectangle(display, (0, 0), (display.shape[1], 100), (0, 0, 0), -1)
                cv2.rectangle(display, (0, 0), (display.shape[1], 100), (0, 255, 0), 2)
                
                cv2.putText(display, f"Sign: {sign_name.upper()}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.putText(display, f"Progress: {sample_count}/{num_samples}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Instructions
                if not self.recording:
                    cv2.putText(display, "Press SPACE to record", (10, display.shape[0]-20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Show mask (hand detection visualization)
                mask_display = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                mask_display = cv2.resize(mask_display, (160, 120))
                display[10:130, display.shape[1]-170:display.shape[1]-10] = mask_display
                
                cv2.imshow('Data Collection', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' ') and not self.recording:
                    if bbox:  # Only record if hand is detected
                        self.recording = True
                        frame_buffer.clear()
                        print(f"Recording sample {sample_count + 1}...")
                    else:
                        print("⚠ No hand detected! Position your hand in frame.")
                elif key == ord('q'):
                    break
                elif key == ord('r') and collected:
                    # Redo last sample
                    collected.pop()
                    sample_count -= 1
                    print(f"Removed last sample. Now at {sample_count}/{num_samples}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        # Save collected data
        if collected:
            self._save_sign_data(sign_name, collected)
            print(f"\n✓ Saved {len(collected)} samples for '{sign_name}'")
        else:
            print("\n⚠ No data collected!")
        
        return len(collected)
    
    def _save_sign_data(self, sign_name, sequences):
        """Save collected sequences"""
        sign_dir = os.path.join(self.data_dir, sign_name)
        os.makedirs(sign_dir, exist_ok=True)
        
        # Load existing data if any
        data_file = os.path.join(sign_dir, "sequences.pkl")
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                existing = pickle.load(f)
            sequences = existing + sequences
        
        # Save
        with open(data_file, 'wb') as f:
            pickle.dump(sequences, f)
        
        # Update stats
        self.stats[sign_name] = len(sequences)
        self.save_stats()
    
    def interactive_collection(self):
        """Interactive mode to collect multiple signs"""
        print("\n" + "="*60)
        print("SIGN LANGUAGE DATA COLLECTION TOOL")
        print("="*60)
        
        # Recommended signs for beginners
        recommended_signs = [
            "HELLO", "THANK_YOU", "PLEASE", "SORRY", 
            "YES", "NO", "HELP", "GOOD", "BAD",
            "WATER", "FOOD", "MORE", "STOP", "GO"
        ]
        
        while True:
            print("\n" + "-"*60)
            print("COLLECTION MENU")
            print("-"*60)
            print("\nCurrent Statistics:")
            if self.stats:
                for sign, count in sorted(self.stats.items()):
                    print(f"  • {sign}: {count} samples")
            else:
                print("  No data collected yet")
            
            print("\nRecommended signs to collect:")
            for i, sign in enumerate(recommended_signs, 1):
                count = self.stats.get(sign, 0)
                status = "✓" if count >= 30 else " "
                print(f"  {status} {i:2d}. {sign} ({count} samples)")
            
            print("\nOptions:")
            print("  1. Collect data for a sign")
            print("  2. View all statistics")
            print("  3. Delete sign data")
            print("  4. Exit")
            
            choice = input("\nYour choice: ").strip()
            
            if choice == "1":
                sign_name = input("\nEnter sign name (e.g., HELLO): ").strip().upper()
                if not sign_name:
                    print("Invalid sign name!")
                    continue
                
                try:
                    num_samples = int(input("Number of samples (recommended 30-50): ").strip())
                except:
                    num_samples = 30
                
                self.collect_sign(sign_name, num_samples)
            
            elif choice == "2":
                self._show_detailed_stats()
            
            elif choice == "3":
                self._delete_sign_data()
            
            elif choice == "4":
                print("\nExiting collection tool...")
                break
    
    def _show_detailed_stats(self):
        """Show detailed statistics"""
        print("\n" + "="*60)
        print("DETAILED STATISTICS")
        print("="*60)
        
        if not self.stats:
            print("No data collected yet!")
            return
        
        total_samples = sum(self.stats.values())
        total_signs = len(self.stats)
        
        print(f"\nTotal Signs: {total_signs}")
        print(f"Total Samples: {total_samples}")
        print(f"Average per Sign: {total_samples/total_signs:.1f}")
        
        print("\nPer-Sign Breakdown:")
        for sign, count in sorted(self.stats.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * (count // 5)
            print(f"  {sign:15s} [{count:3d}] {bar}")
        
        # Check readiness
        ready_signs = [s for s, c in self.stats.items() if c >= 30]
        print(f"\nSigns ready for training (≥30 samples): {len(ready_signs)}/{total_signs}")
    
    def _delete_sign_data(self):
        """Delete data for a specific sign"""
        if not self.stats:
            print("No data to delete!")
            return
        
        print("\nAvailable signs:")
        for i, sign in enumerate(sorted(self.stats.keys()), 1):
            print(f"  {i}. {sign} ({self.stats[sign]} samples)")
        
        sign_name = input("\nEnter sign name to delete: ").strip().upper()
        
        if sign_name not in self.stats:
            print(f"Sign '{sign_name}' not found!")
            return
        
        confirm = input(f"Delete all data for '{sign_name}'? (yes/no): ").strip().lower()
        if confirm == 'yes':
            sign_dir = os.path.join(self.data_dir, sign_name)
            if os.path.exists(sign_dir):
                import shutil
                shutil.rmtree(sign_dir)
                del self.stats[sign_name]
                self.save_stats()
                print(f"✓ Deleted all data for '{sign_name}'")


# ===========================
# 2. DATASET LOADER
# ===========================

def load_dataset(data_dir="sign_data", train_split=0.8):
    """Load collected data and split into train/validation"""
    print("\nLoading dataset...")
    
    X_all = []
    y_all = []
    class_names = []
    
    # Load all sign data
    for sign_name in sorted(os.listdir(data_dir)):
        sign_dir = os.path.join(data_dir, sign_name)
        if not os.path.isdir(sign_dir):
            continue
        
        data_file = os.path.join(sign_dir, "sequences.pkl")
        if not os.path.exists(data_file):
            continue
        
        # Load sequences
        with open(data_file, 'rb') as f:
            sequences = pickle.load(f)
        
        class_idx = len(class_names)
        class_names.append(sign_name)
        
        for seq in sequences:
            X_all.append(seq)
            y_all.append(class_idx)
        
        print(f"  ✓ {sign_name}: {len(sequences)} samples")
    
    if not X_all:
        print("No data found! Please collect data first.")
        return None, None, None, None, None
    
    # Convert to numpy arrays
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    
    # Shuffle
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]
    
    # Split
    split_idx = int(len(X) * train_split)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"\nDataset Summary:")
    print(f"  Classes: {len(class_names)}")
    print(f"  Total samples: {len(X)}")
    print(f"  Training: {len(X_train)}")
    print(f"  Validation: {len(X_val)}")
    
    return X_train, y_train, X_val, y_val, class_names


# ===========================
# 3. MODEL (from original)
# ===========================

class TemporalAttention(layers.Layer):
    def __init__(self, d_model, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % num_heads == 0
        self.depth = d_model // num_heads
        self.wq = layers.Dense(d_model)
        self.wk = layers.Dense(d_model)
        self.wv = layers.Dense(d_model)
        self.dense = layers.Dense(d_model)
    
    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, q, k, v, mask=None):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
        if mask is not None:
            scaled_attention_logits += (mask * -1e9)
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)
        output = tf.transpose(output, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(output, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output, attention_weights
    
    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "num_heads": self.num_heads})
        return config


class SignLanguageTranslator(keras.Model):
    def __init__(self, num_classes, sequence_length=30, feature_dim=81):
        super().__init__()
        self.sequence_length = sequence_length
        self.feature_dim = feature_dim
        self.dense1 = layers.Dense(256, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.3)
        self.dense2 = layers.Dense(128, activation='relu')
        self.bn2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.3)
        self.bilstm1 = layers.Bidirectional(
            layers.LSTM(256, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
        )
        self.bilstm2 = layers.Bidirectional(
            layers.LSTM(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2)
        )
        self.attention = TemporalAttention(d_model=256, num_heads=8)
        self.attention_dropout = layers.Dropout(0.2)
        self.layer_norm = layers.LayerNormalization()
        self.context_dense = layers.Dense(128, activation='relu')
        self.context_dropout = layers.Dropout(0.2)
        self.global_pool = layers.GlobalAveragePooling1D()
        self.final_dense1 = layers.Dense(64, activation='relu')
        self.final_dropout = layers.Dropout(0.3)
        self.output_layer = layers.Dense(num_classes, activation='softmax')
    
    def call(self, inputs, training=False):
        x = self.dense1(inputs)
        x = self.bn1(x, training=training)
        x = self.dropout1(x, training=training)
        x = self.dense2(x)
        x = self.bn2(x, training=training)
        x = self.dropout2(x, training=training)
        x = self.bilstm1(x, training=training)
        x = self.bilstm2(x, training=training)
        attention_output, _ = self.attention(x, x, x)
        attention_output = self.attention_dropout(attention_output, training=training)
        x = self.layer_norm(x + attention_output)
        x = self.context_dense(x)
        x = self.context_dropout(x, training=training)
        x = self.global_pool(x)
        x = self.final_dense1(x)
        x = self.final_dropout(x, training=training)
        output = self.output_layer(x)
        return output


# ===========================
# 4. TRAINING FUNCTION
# ===========================

def train_model(data_dir="sign_data", epochs=50, batch_size=16):
    """Train the model with collected data"""
    
    # Load dataset
    X_train, y_train, X_val, y_val, class_names = load_dataset(data_dir)
    
    if X_train is None:
        return None, None
    
    # Create model
    print("\nBuilding model...")
    model = SignLanguageTranslator(num_classes=len(class_names))
    
    # Build model
    dummy = tf.zeros((1, 30, 81))
    _ = model(dummy, training=False)
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            'best_sign_model.weights.h5',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=True,
            verbose=1
        )
    ]
    
    # Train
    print(f"\nTraining for {epochs} epochs...")
    print("="*60)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save class names
    with open('class_names.json', 'w') as f:
        json.dump(class_names, f)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE!")
    print("="*60)
    print(f"Final Training Accuracy: {history.history['accuracy'][-1]:.2%}")
    print(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]:.2%}")
    print(f"\nModel saved as: best_sign_model.h5")
    print(f"Class names saved as: class_names.json")
    
    return model, class_names


# ===========================
# 5. MAIN MENU
# ===========================

def main():
    print("\n" + "="*60)
    print(" SIGN LANGUAGE ML PIPELINE")
    print("="*60)
    print("\nComplete workflow:")
    print("  1. Collect training data")
    print("  2. Train the model")
    print("  3. Test with webcam")
    print("  4. Exit")
    print("="*60)
    
    while True:
        print("\n" + "-"*60)
        choice = input("\nSelect option (1-4): ").strip()
        
        if choice == "1":
            collector = SignLanguageDataCollector()
            collector.interactive_collection()
        
        elif choice == "2":
            print("\n" + "="*60)
            print("MODEL TRAINING")
            print("="*60)
            
            epochs = input("\nNumber of epochs (default 50): ").strip()
            epochs = int(epochs) if epochs else 50
            
            model, class_names = train_model(epochs=epochs)
            
            if model:
                print("\n✓ Model trained successfully!")
                print("You can now test it with option 3")
        
        elif choice == "3":
            # Check if model exists
            if not os.path.exists('best_sign_model.weights.h5'):
                print("\n⚠ No trained model found!")
                print("Please train the model first (option 2)")
                continue
            
            if not os.path.exists('class_names.json'):
                print("\n⚠ Class names file not found!")
                continue
            
            # Load class names
            with open('class_names.json', 'r') as f:
                class_names = json.load(f)
            
            # Create model and load weights
            print("\nLoading model...")
            model = SignLanguageTranslator(num_classes=len(class_names))
            dummy = tf.zeros((1, 30, 81))
            _ = model(dummy, training=False)
            model.load_weights('best_sign_model.weights.h5')
            
            # Start webcam demo
            print("\nStarting webcam demo...")
            print("Press 'q' to quit")
            
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open webcam")
                continue
            
            # Feature extractor for webcam
            base_model = keras.applications.MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            base_model.trainable = False
            
            feature_model = keras.Sequential([
                base_model,
                layers.GlobalAveragePooling2D(),
                layers.Dense(128, activation='relu'),
                layers.Dense(81)
            ])
            
            frame_buffer = deque(maxlen=30)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            print("\n✓ Webcam started! Show your sign...")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                display = frame.copy()
                
                # Hand detection
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, lower_skin, upper_skin)
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.GaussianBlur(mask, (5, 5), 0)
                
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                hand_detected = False
                if contours:
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    
                    if area > 5000:
                        x, y, w, h = cv2.boundingRect(largest)
                        padding = 20
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2*padding)
                        h = min(frame.shape[0] - y, h + 2*padding)
                        
                        hand_roi = frame[y:y+h, x:x+w]
                        
                        if hand_roi.size > 0:
                            hand_detected = True
                            cv2.rectangle(display, (x, y), (x+w, y+h), (0, 255, 0), 2)
                            
                            # Extract features
                            try:
                                hand_resized = cv2.resize(hand_roi, (224, 224))
                                hand_resized = hand_resized.astype(np.float32) / 255.0
                                hand_resized = np.expand_dims(hand_resized, axis=0)
                                features = feature_model.predict(hand_resized, verbose=0)[0]
                                frame_buffer.append(features)
                            except:
                                pass
                
                # Make prediction
                if len(frame_buffer) == 30:
                    sequence = np.array(list(frame_buffer))
                    sequence = np.expand_dims(sequence, axis=0)
                    
                    prediction = model.predict(sequence, verbose=0)[0]
                    predicted_idx = np.argmax(prediction)
                    confidence = prediction[predicted_idx]
                    
                    if confidence > 0.5:
                        predicted_sign = class_names[predicted_idx]
                        
                        # Display prediction
                        cv2.rectangle(display, (0, 0), (display.shape[1], 80), (0, 0, 0), -1)
                        cv2.putText(display, f"Sign: {predicted_sign}", (10, 40),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                        cv2.putText(display, f"Confidence: {confidence:.1%}", (10, 70),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Status info
                status_color = (0, 255, 0) if hand_detected else (0, 0, 255)
                status_text = "Hand Detected" if hand_detected else "No Hand"
                cv2.putText(display, status_text, (10, display.shape[0]-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
                
                cv2.putText(display, f"Buffer: {len(frame_buffer)}/30", (10, display.shape[0]-50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow('Sign Language Recognition', display)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Webcam demo ended")
        
        elif choice == "4":
            print("\nGoodbye!")
            break
        
        else:
            print("Invalid choice!")


if __name__ == "__main__":
    main()