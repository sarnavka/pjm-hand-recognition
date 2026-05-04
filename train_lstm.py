"""
Dwufazowy trening GRU dla dynamicznych liter PJM.

Faza 1 – Pre-trening na ASL J i Z (data/asl_sequences/)
  GRU backbone uczy się ekstrakcji cech ruchu dłoni.

Faza 2 – Fine-tuning na wszystkich literach PJM (data/pjm_sequences/)
  2a. warmup  – GRU zamrożone, trenuje się tylko nowa głowica klasyfikacyjna
  2b. fine-tune – odblokowanie GRU, douczanie z niskim LR

Wejście:  data/asl_sequences/  + data/pjm_sequences/
Wyjście:  models/pjm_lstm_model.keras
          models/pjm_lstm_meta.pkl  (klasy + scaler)
"""

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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
from sklearn.utils import shuffle as sk_shuffle
import pickle

# ── KONFIGURACJA ──────────────────────────────────────────────
ASL_DIR    = 'data/asl_sequences'
PJM_DIR    = 'data/pjm_sequences'
MODEL_PATH = 'models/pjm_lstm_model.keras'
META_PATH  = 'models/pjm_lstm_meta.pkl'

SEQ_LEN    = 30   # klatek w sekwencji (~1 s przy 30 fps)
N_FEATURES = 63

AUG_TARGET = 200  # minimalna liczba sekwencji na klasę po augmentacji (Faza 2)


# ── AUGMENTACJA SEKWENCJI ─────────────────────────────────────
def augment_to_target(X, y, target=AUG_TARGET, seed=42):
    """
    Dla każdej klasy z mniejszą liczbą próbek dodaje syntetyczne sekwencje:
      - szum gaussowski (σ=0.005)
      - losowe skalowanie amplitudy (×0.90–1.10)
    """
    rng = np.random.default_rng(seed)
    X_out, y_out = [X], [y]

    for cls in np.unique(y):
        idx    = np.where(y == cls)[0]
        needed = max(0, target - len(idx))
        if needed == 0:
            continue

        chosen = rng.choice(idx, size=needed, replace=True)
        seqs   = X[chosen].copy()
        seqs  += rng.normal(0, 0.005, seqs.shape).astype(np.float32)
        scales = rng.uniform(0.90, 1.10, (len(seqs), 1, 1)).astype(np.float32)
        seqs  *= scales

        X_out.append(seqs)
        y_out.append(np.full(needed, cls, dtype=np.int32))

    X_all, y_all = np.vstack(X_out), np.concatenate(y_out)
    return sk_shuffle(X_all, y_all, random_state=seed)


# ── ŁADOWANIE DANYCH ──────────────────────────────────────────
def load_sequences(data_dir):
    """Zwraca (X, y, classes): X shape (N, SEQ_LEN, 63). Pomija puste klasy."""
    all_folders = sorted(os.listdir(data_dir))
    # only keep folders that have at least one .npy file
    classes = [c for c in all_folders
               if os.path.isdir(os.path.join(data_dir, c))
               and any(f.endswith('.npy') for f in os.listdir(os.path.join(data_dir, c)))]
    X, y = [], []
    for i, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        for fname in os.listdir(cls_dir):
            if not fname.endswith('.npy'):
                continue
            seq = np.load(os.path.join(cls_dir, fname))
            if len(seq) != SEQ_LEN:
                idx = np.linspace(0, len(seq) - 1, SEQ_LEN, dtype=int)
                seq = seq[idx]
            X.append(seq)
            y.append(i)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int32), classes


# ── START ──────────────────────────────────────────────────────
print("=" * 60)
print("  DWUFAZOWY TRENING GRU  (ASL pre-train -> PJM fine-tune)")
print("=" * 60)

# ── [1/5] ŁADOWANIE ───────────────────────────────────────────
print("\n[1/5] Ładuję dane...")

X_asl, y_asl, asl_classes = load_sequences(ASL_DIR)
print(f"  ASL:  {len(X_asl)} sekwencji  |  klasy: {asl_classes}")

X_pjm, y_pjm, pjm_classes = load_sequences(PJM_DIR)
n_pjm = len(pjm_classes)
print(f"  PJM:  {len(X_pjm)} sekwencji  |  {n_pjm} klas: {', '.join(pjm_classes)}")

counts = np.bincount(y_pjm)
print(f"  PJM min/max sekwencji na klasę: {counts.min()} / {counts.max()}")

if len(X_asl) == 0:
    print("\n  UWAGA: brak danych ASL – uruchom najpierw generate_asl_sequences.py")
    print("  Kontynuuję tylko z PJM (pominięto Fazę 1).")
    skip_phase1 = True
else:
    skip_phase1 = False

# ── [2/5] NORMALIZACJA ────────────────────────────────────────
print("\n[2/5] Normalizacja...")

# Scaler dopasowany na danych PJM (domain docelowy)
X_pjm_flat = X_pjm.reshape(-1, N_FEATURES)
scaler      = StandardScaler()
scaler.fit(X_pjm_flat)

X_pjm_sc = scaler.transform(X_pjm_flat).reshape(-1, SEQ_LEN, N_FEATURES)

if not skip_phase1:
    X_asl_flat = X_asl.reshape(-1, N_FEATURES)
    X_asl_sc   = scaler.transform(X_asl_flat).reshape(-1, SEQ_LEN, N_FEATURES)

# ── [3/5] BACKBONE ────────────────────────────────────────────
print("\n[3/5] Buduję backbone GRU...")

inp = layers.Input(shape=(SEQ_LEN, N_FEATURES), name='input')
x   = layers.GRU(128, return_sequences=True, name='gru1')(inp)
x   = layers.Dropout(0.3, name='drop1')(x)
x   = layers.GRU(64, name='gru2')(x)
x   = layers.Dropout(0.3, name='drop2')(x)
x   = layers.Dense(64, activation='relu', name='dense_shared')(x)
x   = layers.Dropout(0.25, name='drop3')(x)
backbone_out = x

backbone = Model(inp, backbone_out, name='backbone')
print(f"  Backbone: {backbone.count_params():,} parametrów")

# ── FAZA 1: ASL PRE-TRAIN ─────────────────────────────────────
if not skip_phase1:
    print("\n── FAZA 1: Pre-trening na ASL J i Z ──────────────────────")

    n_asl_cls = len(asl_classes)
    out1  = layers.Dense(n_asl_cls, activation='softmax', name='head_asl')(backbone.output)
    model1 = Model(backbone.input, out1, name='model_asl')
    model1.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    X1_tr, X1_val, y1_tr, y1_val = train_test_split(
        X_asl_sc, y_asl, test_size=0.2, stratify=y_asl, random_state=42
    )
    print(f"  Trening ASL: {len(X1_tr)}  Walidacja: {len(X1_val)}")

    cb1 = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=10,
            restore_best_weights=True, verbose=0
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy', factor=0.5,
            patience=5, min_lr=1e-7, verbose=0
        ),
    ]

    h1 = model1.fit(
        X1_tr, y1_tr,
        validation_data=(X1_val, y1_val),
        epochs=40,
        batch_size=64,
        callbacks=cb1,
        verbose=1
    )
    asl_acc = max(h1.history['val_accuracy']) * 100
    print(f"  ASL val accuracy: {asl_acc:.1f}%  (backbone gotowy)")

# ── FAZA 2a: PJM WARMUP (GRU zamrożone) ──────────────────────
print("\n── FAZA 2a: PJM warmup (GRU zamrożone) ───────────────────")

X_pjm_aug, y_pjm_aug = augment_to_target(X_pjm_sc, y_pjm)
print(f"  PJM po augmentacji: {len(X_pjm_aug)} sekwencji "
      f"(cel {AUG_TARGET}/klasę × {n_pjm} klas)")

X2_tr, X2_val, y2_tr, y2_val = train_test_split(
    X_pjm_aug, y_pjm_aug, test_size=0.2, stratify=y_pjm_aug, random_state=42
)
print(f"  Trening PJM: {len(X2_tr)}  Walidacja: {len(X2_val)}")

# Zamroź GRU do warmup
for layer in backbone.layers:
    if isinstance(layer, layers.GRU):
        layer.trainable = False

out2   = layers.Dense(n_pjm, activation='softmax', name='head_pjm')(backbone.output)
model2 = Model(backbone.input, out2, name='model_pjm')
model2.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"  Trainable params (warmup): {model2.count_params():,}")

cb_warm = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', patience=8,
        restore_best_weights=True, verbose=0
    ),
]

model2.fit(
    X2_tr, y2_tr,
    validation_data=(X2_val, y2_val),
    epochs=20,
    batch_size=32,
    callbacks=cb_warm,
    verbose=1
)

# ── FAZA 2b: PJM FINE-TUNE (odblokowanie GRU) ─────────────────
print("\n── FAZA 2b: PJM fine-tune (pełny model) ──────────────────")

for layer in backbone.layers:
    layer.trainable = True

model2.compile(
    optimizer=tf.keras.optimizers.Adam(5e-5),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
print(f"  Trainable params (fine-tune): {model2.count_params():,}")

cb_ft = [
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

history = model2.fit(
    X2_tr, y2_tr,
    validation_data=(X2_val, y2_val),
    epochs=100,
    batch_size=32,
    callbacks=cb_ft,
    verbose=1
)

# ── [4/5] EWALUACJA ───────────────────────────────────────────
print("\n[4/5] Ewaluacja...")
y_pred = np.argmax(model2.predict(X2_val, verbose=0), axis=1)
acc    = np.mean(y_pred == y2_val) * 100

print()
print("=" * 60)
print("  WYNIKI")
print("=" * 60)
print(f"  Val accuracy: {acc:.2f}%")
print()
print(f"  {'Klasa':<6}  {'Recall':>8}  {'Próbek':>8}")
print(f"  {'-' * 26}")
for i, cls in enumerate(pjm_classes):
    mask = y2_val == i
    if mask.sum() == 0:
        continue
    rec = (y_pred[mask] == i).mean() * 100
    print(f"  {cls:<6}  {rec:>7.1f}%  {mask.sum():>8}")

# ── [5/5] ZAPIS ───────────────────────────────────────────────
print("\n[5/5] Zapisuję model i metadane...")
os.makedirs('models', exist_ok=True)

with open(META_PATH, 'wb') as f:
    pickle.dump({'classes': pjm_classes, 'scaler': scaler}, f)

print()
print(f"  Model:     {MODEL_PATH}")
print(f"  Metadane:  {META_PATH}")
print("=" * 60)
print("  GOTOWE → uruchom app.py")
print("=" * 60)
