"""
Trening klasyfikatora na landmarkach dloni (PJM).

Wejscie:  data/pjm_landmarks.npz  (63 cechy = 21 punktow x 3)
Wyjscie:  models/pjm_landmark_model.keras

Architektura: MLP z BatchNorm + Dropout
Czas treningu: ~2-3 minuty
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle

DATA_PATH  = 'data/pjm_landmarks.npz'
ASL_PATH   = 'data/asl_landmarks.npz'   # opcjonalne – generuje extract_asl_landmarks.py
MODEL_PATH = 'models/pjm_landmark_model.keras'
SCALER_PATH = 'models/pjm_landmark_scaler.pkl'

print("=" * 60)
print("  TRENING KLASYFIKATORA LANDMARKOW PJM")
print("=" * 60)

# ── Dane ──────────────────────────────────────────────────────
print("\n[1/4] Laduje dane...")
data    = np.load(DATA_PATH, allow_pickle=True)
X       = data['X']
y       = data['y']
classes = list(data['classes'])
n_cls   = len(classes)

print(f"  Probek PJM: {len(X)}")
print(f"  Cechy:      {X.shape[1]}  (21 landmarkow x 3 wspolrzedne)")
print(f"  Klas:       {n_cls}")

# Opcjonalne: domieszaj dane ASL dla podobnych liter (B,C,E,G,K,O,T,U,W,Y)
if os.path.exists(ASL_PATH):
    asl_data = np.load(ASL_PATH, allow_pickle=True)
    X = np.vstack([X, asl_data['X']])
    y = np.concatenate([y, asl_data['y']])
    n_asl = len(asl_data['X'])
    print(f"  + ASL data: {n_asl} probek  (litery podobne: B,C,E,G,K,O,T,U,W,Y)")
    print(f"  Lacznie:    {len(X)} probek")
else:
    print(f"  ASL data: brak (uruchom extract_asl_landmarks.py aby dodac)")

# Rozklad klas
counts = np.bincount(y, minlength=n_cls)
print(f"  Min probek na klase: {counts.min()}")
print(f"  Max probek na klase: {counts.max()}")

# ── Normalizacja ──────────────────────────────────────────────
print("\n[2/4] Normalizacja + split...")
scaler  = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, stratify=y, random_state=42
)
print(f"  Trening:   {len(X_train)}")
print(f"  Walidacja: {len(X_val)}")

class_weight = {i: 1.0 for i in range(len(classes))}

# ── Model MLP ─────────────────────────────────────────────────
print("\n[3/4] Buduje i trenuje model...")

inp = layers.Input(shape=(X.shape[1],))
x   = layers.Dense(512, activation='relu')(inp)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.4)(x)
x   = layers.Dense(256, activation='relu')(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.35)(x)
x   = layers.Dense(128, activation='relu')(x)
x   = layers.BatchNormalization()(x)
x   = layers.Dropout(0.3)(x)
out = layers.Dense(n_cls, activation='softmax')(x)

model = Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"  Parametry: {model.count_params():,}")

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=15,
        restore_best_weights=True, verbose=0
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_accuracy', factor=0.5,
        patience=7, min_lr=1e-7, verbose=0
    ),
    tf.keras.callbacks.ModelCheckpoint(
        MODEL_PATH, monitor='val_accuracy',
        save_best_only=True, verbose=0
    ),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=150,
    batch_size=64,
    class_weight=class_weight,
    callbacks=callbacks,
    verbose=1
)

# ── Wyniki ────────────────────────────────────────────────────
print("\n[4/4] Ewaluacja...")
y_pred = np.argmax(model.predict(X_val, verbose=0), axis=1)
acc    = np.mean(y_pred == y_val) * 100

print()
print("=" * 60)
print("  WYNIKI")
print("=" * 60)
print(f"  Val accuracy: {acc:.2f}%")
print()

# Per-class recall dla trudnych liter
hard_cls = ['L', 'Ł', 'J', 'I', 'CH', 'CZ', 'O', 'Ó',
            'E', 'Ę', 'S', 'Ś', 'C', 'Ć']
print(f"  {'Klasa':<8} {'Recall':>8}  {'Probek':>8}")
print(f"  {'-'*28}")
for cls in hard_cls:
    if cls not in classes:
        continue
    i    = classes.index(cls)
    mask = y_val == i
    if mask.sum() == 0:
        continue
    rec  = (y_pred[mask] == i).mean() * 100
    print(f"  {cls:<8} {rec:>7.1f}%  {mask.sum():>8}")

# Zapis scalera
with open(SCALER_PATH, 'wb') as f:
    pickle.dump(scaler, f)

print()
print(f"  Model:  {MODEL_PATH}")
print(f"  Scaler: {SCALER_PATH}")
print("=" * 60)
print("  GOTOWE -> zaktualizuj app.py o tryb landmarkow")
print("=" * 60)
