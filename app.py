"""
Rozpoznawanie liter PJM w czasie rzeczywistym.

Tryby:
  L  – Landmarki PJM  (MLP + GRU, zalecany)
  P  – CNN PJM        (obraz)
  A  – CNN ASL        (obraz)

Sterowanie:
  L / P / A  – zmień tryb
  S          – zapisz screenshot
  Q / Esc    – wyjdź
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import cv2
import numpy as np
import pickle
from collections import deque
from datetime import datetime

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from finger_features import compute_finger_features

# ── PIL – polskie znaki ────────────────────────────────────────
try:
    from PIL import ImageFont, ImageDraw, Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_font_cache: dict = {}

def _get_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    for path in [r"C:\Windows\Fonts\arial.ttf",
                 r"C:\Windows\Fonts\calibri.ttf",
                 r"C:\Windows\Fonts\segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(path):
            try:
                _font_cache[size] = ImageFont.truetype(path, size)
                return _font_cache[size]
            except Exception:
                pass
    _font_cache[size] = ImageFont.load_default()
    return _font_cache[size]

def draw_text_pl(frame, text, pos, size, color_bgr, stroke=0):
    """Rysuje tekst z polskimi znakami. pos = lewy-górny róg."""
    if not _PIL_OK:
        cv2.putText(frame, text, (pos[0], pos[1] + size),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 28, color_bgr, 1)
        return
    font = _get_font(size)
    img  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    rgb  = (color_bgr[2], color_bgr[1], color_bgr[0])
    kw   = {'stroke_width': stroke, 'stroke_fill': (0, 0, 0)} if stroke else {}
    draw.text(pos, text, font=font, fill=rgb, **kw)
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_big_letter(frame, text, color_bgr):
    """Wyśrodkowana duża litera – obsługuje polskie znaki."""
    h, w = frame.shape[:2]
    if not _PIL_OK:
        fs = 5
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, 9)[0]
        cx = (w - ts[0]) // 2; cy = h // 2 + ts[1] // 2 + 30
        cv2.putText(frame, text, (cx+3, cy+3), cv2.FONT_HERSHEY_SIMPLEX, fs, (0,0,0), 13)
        cv2.putText(frame, text, (cx, cy),     cv2.FONT_HERSHEY_SIMPLEX, fs, color_bgr, 9)
        return
    sz   = min(h // 3, 190)
    font = _get_font(sz)
    img  = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    bb   = font.getbbox(text)
    cx   = (w - (bb[2] - bb[0])) // 2 - bb[0]
    cy   = (h - (bb[3] - bb[1])) // 2 - bb[1] + 30
    draw.text((cx+3, cy+3), text, font=font, fill=(0, 0, 0),
              stroke_width=4, stroke_fill=(0, 0, 0))
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((cx, cy), text, font=font, fill=rgb,
              stroke_width=2, stroke_fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


# ── KONFIGURACJA ──────────────────────────────────────────────
SEQ_LEN        = 30      # klatek w sekwencji GRU
GRU_THRESHOLD  = 0.70    # min. pewność GRU żeby nadpisać MLP
MLP_SMOOTH     = 20      # klatek do wygładzania MLP
CONF_THRESHOLD = 0.55    # poniżej = "niska pewność"
LOCK_FRAMES    = 18      # klatek tej samej litery → zablokuj wyświetlanie
LOCK_CONF      = 0.60    # min. pewność żeby kandydat mógł zablokować
MOTION_GATE    = 0.012   # min. średnia prędkość landmarków w buforze → aktywuje GRU
STILL_THRESH   = 0.005   # prędkość poniżej tej = klatka "nieruchoma"
STILL_RESET_N  = 20      # tyle klatek nieruchomości → wyczyść bufor GRU
FREEZE_FRAMES  = 35      # tyle klatek trzymaj wynik GRU po wykryciu (~1.2 s)
GESTURE_START_VEL  = MOTION_GATE  # prędkość startująca nagrywanie gestu
GESTURE_END_FRAMES = 12           # klatek bezruchu → gest zakończony
CIRCULARITY_THRESH = 0.28         # max std/mean promieni → ruch okrągły
IMG_SIZE       = 224
SCREENSHOT_DIR = 'screenshots'
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

DYNAMIC_LETTERS = {
    'Ą', 'Ć', 'CH', 'CZ', 'D', 'Ę', 'F', 'G', 'H',
    'J', 'K', 'Ł', 'Ń', 'Ó', 'RZ', 'Ś', 'SZ', 'Z', 'Ź', 'Ż',
}

SHAPE_MOTION_MAP = {}

# Wymagany typ ruchu dla liter klasyfikowanych przez GRU
# Jeśli wykryty ruch nie pasuje → wynik GRU odrzucany
MOTION_REQUIRED = {
    'D':  'circular',
    'E':  'circular',
    'J':  'circular',
    'Ż':  'circular',
    'CH': 'down',
    'N':  'down',
    'Ń':  'down',
    'L':  'directional',
    'SZ': 'directional',
    'Z':  'directional',
    'CZ': 'directional',
    'RZ': 'directional',
    'Ź':  'directional',
}

COLOR_MLP  = (200, 80,  200)   # fioletowy – MLP / statyczne
COLOR_GRU  = (80,  200, 80)    # zielony   – GRU / dynamiczne
COLOR_CNN  = (50,  120, 255)   # niebieski – CNN
COLOR_WARN = (0,   165, 255)   # pomarańczowy – niska pewność


# ── ŁADOWANIE MODELI ──────────────────────────────────────────
print("=" * 60)
print("  Ładuję modele...")
print("=" * 60)

# CNN
cnn_models  = {}
cnn_classes = {}
for name, model_path, data_dir in [
    ('PJM', 'models/pjm_model_v2_best.h5',
             'data/pjm_dataset'),
    ('ASL', 'models/asl_model.h5',
             'data/asl_alphabet_train/asl_alphabet_train'),
]:
    if not os.path.exists(model_path):
        continue
    try:
        m = tf.keras.models.load_model(model_path, compile=False)
        cnn_models[name] = m
        if os.path.exists(data_dir):
            gen = ImageDataGenerator().flow_from_directory(
                data_dir, target_size=(IMG_SIZE, IMG_SIZE),
                batch_size=1, class_mode='categorical', shuffle=False)
            cnn_classes[name] = [k for k, _ in sorted(
                gen.class_indices.items(), key=lambda x: x[1])]
        else:
            cnn_classes[name] = [str(i) for i in range(m.output.shape[-1])]
        print(f"  CNN {name}: OK  ({m.output.shape[-1]} klas)")
    except Exception as e:
        print(f"  CNN {name}: błąd – {e}")

# MLP + MediaPipe
lm_model = lm_scaler = lm_classes = lm_detector = None
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    need = ['models/pjm_landmark_model.keras',
            'models/pjm_landmark_scaler.pkl',
            'models/hand_landmarker.task']
    if all(os.path.exists(p) for p in need):
        lm_model = tf.keras.models.load_model(need[0], compile=False)
        with open(need[1], 'rb') as f:
            lm_scaler = pickle.load(f)
        gen = ImageDataGenerator().flow_from_directory(
            'data/pjm_dataset', target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=1, class_mode='categorical', shuffle=False)
        lm_classes = [k for k, _ in sorted(
            gen.class_indices.items(), key=lambda x: x[1])]
        base_opts = mp_python.BaseOptions(model_asset_path=need[2])
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base_opts, num_hands=1,
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE)
        lm_detector = mp_vision.HandLandmarker.create_from_options(opts)
        print(f"  MLP: OK  ({lm_model.output.shape[-1]} klas)")
        print(f"  MediaPipe: OK")
    else:
        print("  MLP: brak plików")
except Exception as e:
    print(f"  MLP: niedostępny ({e})")

# GRU
gru_model = gru_classes = gru_scaler = None
if all(os.path.exists(p) for p in ['models/pjm_lstm_model.keras',
                                    'models/pjm_lstm_meta.pkl']):
    try:
        gru_model = tf.keras.models.load_model(
            'models/pjm_lstm_model.keras', compile=False)
        with open('models/pjm_lstm_meta.pkl', 'rb') as f:
            meta = pickle.load(f)
        gru_classes = list(meta['classes'])
        gru_scaler  = meta['scaler']
        print(f"  GRU: OK  ({len(gru_classes)} klas dynamicznych)")
    except Exception as e:
        print(f"  GRU: błąd – {e}")

print("=" * 60)

if not cnn_models and not lm_detector:
    print("Brak modeli – uruchom najpierw trening!"); exit(1)

MODE = 'LANDMARKS' if lm_detector else list(cnn_models.keys())[0]


# ── BUFORY I STAN RUCHU ───────────────────────────────────────
mlp_history  = deque(maxlen=MLP_SMOOTH)  # wygładzanie MLP
seq_buffer   = deque(maxlen=SEQ_LEN)     # sekwencja dla GRU
trail_pts    = deque(maxlen=30)          # ślad czubka palca

prev_vec63   = None    # poprzednia klatka do obliczenia prędkości
still_frames = 0       # licznik klatek bez ruchu
smooth_vel   = 0.0     # wygładzona prędkość (EMA)

frozen_letter = ''     # zamrożony wynik GRU
frozen_conf   = 0.0
frozen_count  = 0      # ile klatek jeszcze trzymać zamrożony wynik

hide_letter = ''   # litera pokazana po schowaniu dłoni
hide_conf   = 0.0
hide_count  = 0    # ile klatek jeszcze pokazywać wynik po schowaniu dłoni
HIDE_FREEZE = 90   # ~3 s przy 30 fps

lock_letter = ''   # aktualnie zablokowana (stabilna) litera
lock_cand   = ''   # kandydat próbujący zastąpić lock_letter
lock_count  = 0    # ile klatek kandydat wygrywa z rzędu

gesture_state  = 'IDLE'            # 'IDLE' | 'MOVING'
gesture_buffer = deque(maxlen=90)  # landmarki bieżącego gestu (maks ~3 s)
gesture_still  = 0                 # klatek bezruchu po fazie MOVING
motion_type    = 'none'            # 'circular' | 'down' | 'up' | 'directional' | 'none'


# ── DETEKCJA RĘKI ─────────────────────────────────────────────
def detect_hand(frame_bgr):
    """Zwraca (lm_list, vec63, vec82, bbox) lub (None,...) gdy brak dłoni."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = lm_detector.detect(mp_img)
    if not result.hand_landmarks:
        return None, None, None, None
    lm    = result.hand_landmarks[0]
    wrist = lm[0]
    pts   = np.array([[p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]
                       for p in lm], dtype=np.float32)
    vec63 = pts.flatten()
    vec82 = np.concatenate([vec63, compute_finger_features(pts)])
    fh, fw = frame_bgr.shape[:2]
    xs = [int(p.x * fw) for p in lm]
    ys = [int(p.y * fh) for p in lm]
    bbox = (max(0, min(xs)-20), max(0, min(ys)-20),
            min(fw, max(xs)+20), min(fh, max(ys)+20))
    return lm, vec63, vec82, bbox


def draw_skeleton(frame, lm_list):
    fh, fw = frame.shape[:2]
    conns = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)]
    for a, b in conns:
        ax = int(lm_list[a].x * fw); ay = int(lm_list[a].y * fh)
        bx = int(lm_list[b].x * fw); by = int(lm_list[b].y * fh)
        cv2.line(frame, (ax, ay), (bx, by), COLOR_MLP, 2)
    for p in lm_list:
        cv2.circle(frame, (int(p.x * fw), int(p.y * fh)), 4, (220, 120, 255), -1)


# ── RUCH ─────────────────────────────────────────────────────
def frame_velocity(v_new, v_old):
    """Średnia prędkość 21 landmarków między dwoma klatkami (znorm. wsp.)."""
    return float(np.mean(np.linalg.norm(
        (v_new - v_old).reshape(21, 3), axis=1)))

def buffer_motion(buf):
    """Średnia prędkość przez cały bufor – miara ile ruchu jest w sekwencji."""
    if len(buf) < 2:
        return 0.0
    arr = np.array(buf)
    vel = np.diff(arr, axis=0).reshape(-1, 21, 3)
    return float(np.mean(np.linalg.norm(vel, axis=2)))


# ── PREDYKCJA ─────────────────────────────────────────────────
def mlp_predict(vec82):
    """MLP – jeden kadr, 82 cechy, zwraca probs nad lm_classes."""
    v = lm_scaler.transform(vec82.reshape(1, -1))
    return lm_model.predict(v, verbose=0)[0]


def gru_predict():
    """GRU – ostatnie SEQ_LEN klatek z seq_buffer, zwraca (letter, conf)."""
    if gru_model is None or len(seq_buffer) < SEQ_LEN:
        return None, 0.0
    arr   = np.array(seq_buffer, dtype=np.float32)          # (SEQ_LEN, 63)
    flat  = gru_scaler.transform(arr.reshape(-1, 63))
    seq   = flat.reshape(1, SEQ_LEN, 63)
    probs = gru_model.predict(seq, verbose=0)[0]
    idx   = int(probs.argmax())
    return gru_classes[idx], float(probs[idx])


def classify_motion_type(trail):
    """Klasyfikuje typ ruchu na podstawie śladów czubka palca (piksele)."""
    pts = list(trail)
    if len(pts) < 6:
        return 'none'
    arr    = np.array(pts, dtype=np.float32)
    center = arr.mean(axis=0)
    radii  = np.linalg.norm(arr - center, axis=1)
    mean_r = radii.mean()
    if mean_r < 8:
        return 'none'
    # Okrągłość: niskie odch. std promieni → trajektoria bliska okręgowi
    circularity = radii.std() / mean_r
    if circularity < CIRCULARITY_THRESH:
        return 'circular'
    dy   = float(arr[-1, 1] - arr[0, 1])   # > 0 = w dół (układ ekranu)
    dx   = float(arr[-1, 0] - arr[0, 0])
    span = float(max(arr[:, 0].max() - arr[:, 0].min(),
                     arr[:, 1].max() - arr[:, 1].min()))
    if span < 12:
        return 'none'
    if abs(dy) > abs(dx) * 1.2:
        return 'down' if dy > 0 else 'up'
    return 'directional'


def gru_predict_buffer(buf):
    """GRU na segmencie gestu – resampluje do SEQ_LEN klatek."""
    if gru_model is None or len(buf) < 8:
        return None, 0.0
    arr = np.array(buf, dtype=np.float32)
    if len(arr) >= SEQ_LEN:
        idx = np.linspace(0, len(arr) - 1, SEQ_LEN, dtype=int)
        arr = arr[idx]
    else:
        pad = np.tile(arr[-1], (SEQ_LEN - len(arr), 1))
        arr = np.vstack([arr, pad])
    flat  = gru_scaler.transform(arr.reshape(-1, 63))
    seq   = flat.reshape(1, SEQ_LEN, 63)
    probs = gru_model.predict(seq, verbose=0)[0]
    idx   = int(probs.argmax())
    return gru_classes[idx], float(probs[idx])


def fuse(mlp_probs, cur_motion_type='none'):
    """
    Zwraca (letter, conf, source).
    Priorytet: zamrożony GRU (z segmentacji gestu) > MLP.
    MLP jest tłumiony gdy wykryto ruch (żeby nie mieszać statycznych liter).
    """
    global frozen_count
    # Zamrożony wynik GRU z zakończonego gestu
    if frozen_count > 0:
        frozen_count -= 1
        return frozen_letter, frozen_conf, 'GRU'
    # MLP – wygładzony wynik
    if mlp_probs is not None and lm_classes:
        avg    = np.mean(list(mlp_history), axis=0) if mlp_history else mlp_probs
        i      = int(avg.argmax())
        letter = lm_classes[i]
        conf   = float(avg[i])
        # Tłum statyczne litery podczas aktywnego ruchu (Pomysł 2)
        if cur_motion_type != 'none' and letter not in DYNAMIC_LETTERS:
            conf *= 0.5
        return letter, conf, 'MLP'
    return '', 0.0, ''


# ── CNN ───────────────────────────────────────────────────────
def cnn_predict(roi_bgr):
    roi = cv2.cvtColor(cv2.resize(roi_bgr, (IMG_SIZE, IMG_SIZE)),
                       cv2.COLOR_BGR2RGB)
    inp = preprocess_input(np.expand_dims(roi.astype(np.float32), 0))
    return cnn_models[MODE].predict(inp, verbose=0)[0]


# ── UI ────────────────────────────────────────────────────────
def draw_trail(frame):
    pts = list(trail_pts)
    n   = len(pts)
    for i in range(1, n):
        alpha = i / n
        c = (int(80 * alpha), int(200 * alpha), int(80 * alpha))
        cv2.line(frame, pts[i-1], pts[i], c, max(1, int(alpha * 3)))


def draw_ui(frame, letter, conf, source, bbox=None):
    h, w = frame.shape[:2]

    # Ślad ruchu (wyłączone)
    # draw_trail(frame)

    # Szkielet dłoni (bbox dla CNN)
    if bbox and MODE != 'LANDMARKS':
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      COLOR_CNN, 2)

    # Panel górny
    cv2.rectangle(frame, (0, 0), (w, 58), (18, 18, 18), -1)
    mode_str = {'LANDMARKS': 'Landmark PJM  (MLP+GRU)',
                'PJM': 'CNN PJM', 'ASL': 'CNN ASL'}.get(MODE, MODE)
    color_mode = COLOR_MLP if MODE == 'LANDMARKS' else COLOR_CNN
    cv2.putText(frame, f"Tryb: {mode_str}",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color_mode, 2)
    cv2.putText(frame, "L=Landmark  P=PJM  A=ASL  S=screen  Q=wyjdz",
                (w - 490, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

    if not letter:
        if gesture_state == 'MOVING':
            msg = "Analizuje..."
            col = (0, 160, 255)
        else:
            msg = "Pokaz dlon do kamery"
            col = (120, 120, 120)
        ts  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(frame, msg, ((w - ts[0]) // 2, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        return

    # Kolor litery
    if source == 'HIDE':
        lc = (0, 200, 255)   # żółty – wynik po schowaniu dłoni
    elif source == 'GRU':
        lc = COLOR_GRU
    elif conf >= CONF_THRESHOLD:
        lc = COLOR_MLP if MODE == 'LANDMARKS' else COLOR_CNN
    else:
        lc = COLOR_WARN

    # Duża litera
    draw_big_letter(frame, letter, lc)

    # Banner "ROZPOZNANO" gdy dłoń schowana
    if source == 'HIDE':
        msg = "ROZPOZNANO:"
        ts  = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        cv2.putText(frame, msg, ((w - ts[0]) // 2, h // 2 - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, lc, 2)

    # Pasek pewności
    bw = 320; bx = (w - bw) // 2; by = h - 75
    cv2.rectangle(frame, (bx, by), (bx + bw, by + 22), (50, 50, 50), -1)
    cv2.rectangle(frame, (bx, by), (bx + int(bw * conf), by + 22), lc, -1)
    cv2.putText(frame, f"{conf*100:.0f}%",
                (bx + bw + 8, by + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # Etykieta źródła (GRU / MLP)
    if MODE == 'LANDMARKS':
        src_color = COLOR_GRU if source == 'GRU' else COLOR_MLP
        src_label = f"[{source}]  {'dynamiczna' if letter in DYNAMIC_LETTERS else 'statyczna'}"
        draw_text_pl(frame, src_label, (bx, by - 26), 16, src_color)

    # Niska pewność
    if conf < CONF_THRESHOLD:
        cv2.putText(frame, "niska pewnosc",
                    (bx, by + 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_WARN, 1)

    # Status gestu i ruchu
    if MODE == 'LANDMARKS' and gru_model is not None:
        bw2 = 160; bx2 = w - bw2 - 15
        # Pasek segmentacji gestu
        by2    = h - 50
        g_fill = min(len(gesture_buffer) / max(SEQ_LEN, 1), 1.0)
        gs_col = COLOR_GRU if gesture_state == 'MOVING' else (80, 80, 160)
        cv2.rectangle(frame, (bx2, by2), (bx2 + bw2, by2 + 8), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx2, by2), (bx2 + int(bw2 * g_fill), by2 + 8), gs_col, -1)
        gs_label = 'RUCH' if gesture_state == 'MOVING' else 'CZEKA'
        cv2.putText(frame, f"gest: {gs_label}",
                    (bx2, by2 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gs_col, 1)
        # Pasek ruchu (prędkość)
        by3      = h - 28
        vel_fill = min(smooth_vel / (MOTION_GATE * 3), 1.0)
        vel_col  = (60, 220, 80) if smooth_vel >= MOTION_GATE else (80, 80, 80)
        cv2.rectangle(frame, (bx2, by3), (bx2 + bw2, by3 + 8), (40, 40, 40), -1)
        cv2.rectangle(frame, (bx2, by3), (bx2 + int(bw2 * vel_fill), by3 + 8), vel_col, -1)
        cv2.putText(frame, "ruch",
                    (bx2, by3 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1)
        # Typ ruchu z ostatniego gestu (Pomysł 2)
        if motion_type != 'none':
            mt_map  = {'circular': 'okragly', 'down': 'w dol',
                       'up': 'w gore', 'directional': 'kierunkowy'}
            mt_text = mt_map.get(motion_type, motion_type)
            cv2.putText(frame, mt_text,
                        (bx2, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 60), 1)


# ── KAMERA ────────────────────────────────────────────────────
print("\nUruchamiam kamerę...")
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Błąd: brak kamery!"); exit(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print("Gotowe!\n")

# ── GŁÓWNA PĘTLA ──────────────────────────────────────────────
letter = ''
conf   = 0.0
source = ''

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame  = cv2.flip(frame, 1)
    fh, fw = frame.shape[:2]

    bbox       = None
    gru_letter = None
    gru_conf   = 0.0

    if MODE == 'LANDMARKS' and lm_detector:
        lm_list, vec63, vec82, bbox = detect_hand(frame)

        if lm_list is not None:
            draw_skeleton(frame, lm_list)

            # Ślad czubka palca wskazującego
            tip = lm_list[8]
            trail_pts.append((int(tip.x * fw), int(tip.y * fh)))

            # MLP
            mlp_probs = mlp_predict(vec82)
            mlp_history.append(mlp_probs)

            # Prędkość ręki między klatkami
            vel = frame_velocity(vec63, prev_vec63) if prev_vec63 is not None else 0.0
            prev_vec63 = vec63
            smooth_vel = 0.65 * smooth_vel + 0.35 * vel

            # Rolling bufor do wizualizacji paska
            if smooth_vel < STILL_THRESH:
                still_frames += 1
                if still_frames >= STILL_RESET_N:
                    seq_buffer.clear()
                    still_frames = 0
            else:
                still_frames = 0
            seq_buffer.append(vec63)

            # ── Maszyna stanów gestu – segmentacja (Pomysł 3) ────
            if gesture_state == 'IDLE':
                if smooth_vel >= GESTURE_START_VEL:
                    gesture_state = 'MOVING'
                    gesture_buffer.clear()
                    gesture_still = 0
            elif gesture_state == 'MOVING':
                gesture_buffer.append(vec63)
                if smooth_vel < STILL_THRESH:
                    gesture_still += 1
                    if gesture_still >= GESTURE_END_FRAMES:
                        # Gest zakończony – klasyfikuj typ ruchu i literę
                        motion_type = classify_motion_type(trail_pts)

                        # 1. Fuzja kształt MLP + ruch (priorytet przed GRU)
                        fused = None
                        if mlp_probs is not None and lm_classes and mlp_history:
                            avg_shape = np.mean(list(mlp_history), axis=0)
                            mlp_shape = lm_classes[int(avg_shape.argmax())]
                            fused = SHAPE_MOTION_MAP.get((mlp_shape, motion_type))

                        if fused:
                            frozen_letter = fused
                            frozen_conf   = 0.92
                            frozen_count  = FREEZE_FRAMES
                        elif gru_model is not None:
                            # 2. GRU – z filtrem wymaganego ruchu
                            g_letter, g_conf = gru_predict_buffer(list(gesture_buffer))
                            req_motion = MOTION_REQUIRED.get(g_letter)
                            motion_ok  = (motion_type != 'none'
                                          and (req_motion is None
                                               or motion_type == req_motion))
                            if (g_letter is not None
                                    and g_letter in DYNAMIC_LETTERS
                                    and g_conf >= GRU_THRESHOLD
                                    and motion_ok):
                                frozen_letter = g_letter
                                frozen_conf   = g_conf
                                frozen_count  = FREEZE_FRAMES
                        gesture_state = 'IDLE'
                        gesture_buffer.clear()
                        gesture_still = 0
                else:
                    gesture_still = 0

            # Fuzja: tłum MLP tylko gdy gest aktywny (Pomysł 2)
            letter, conf, source = fuse(
                mlp_probs,
                motion_type if gesture_state == 'MOVING' else 'none'
            )
        else:
            # Dłoń znikła – dokończ analizę gestu jeśli był w trakcie
            if gesture_state == 'MOVING' and len(gesture_buffer) >= 8:
                motion_type = classify_motion_type(trail_pts)
                fused = None
                if mlp_probs is not None and lm_classes and mlp_history:
                    avg_shape = np.mean(list(mlp_history), axis=0)
                    mlp_shape = lm_classes[int(avg_shape.argmax())]
                    fused = SHAPE_MOTION_MAP.get((mlp_shape, motion_type))
                if fused:
                    frozen_letter = fused
                    frozen_conf   = 0.92
                    frozen_count  = FREEZE_FRAMES
                elif gru_model is not None:
                    g_letter, g_conf = gru_predict_buffer(list(gesture_buffer))
                    req_motion = MOTION_REQUIRED.get(g_letter)
                    motion_ok  = (motion_type != 'none'
                                  and (req_motion is None
                                       or motion_type == req_motion))
                    if (g_letter is not None
                            and g_letter in DYNAMIC_LETTERS
                            and g_conf >= GRU_THRESHOLD
                            and motion_ok):
                        frozen_letter = g_letter
                        frozen_conf   = g_conf
                        frozen_count  = FREEZE_FRAMES

            # Zapamiętaj ostatni pewny wynik (statyczny lub właśnie zamrożony GRU)
            if frozen_count > 0 and hide_count == 0:
                hide_letter = frozen_letter
                hide_conf   = frozen_conf
                hide_count  = HIDE_FREEZE
            elif letter and conf >= CONF_THRESHOLD and hide_count == 0:
                hide_letter = letter
                hide_conf   = conf
                hide_count  = HIDE_FREEZE

            mlp_history.clear()
            seq_buffer.clear()
            trail_pts.clear()
            prev_vec63 = None
            still_frames = 0
            smooth_vel = 0.0
            gesture_state = 'IDLE'
            gesture_buffer.clear()
            gesture_still = 0
            motion_type = 'none'
            letter, conf, source = '', 0.0, ''

    elif MODE in cnn_models and lm_detector:
        # CNN z auto-detekcją dłoni
        lm_list, _, _, bbox = detect_hand(frame.copy())
        if lm_list is not None and bbox:
            x1, y1, x2, y2 = bbox
            if x2 > x1 and y2 > y1:
                probs = cnn_predict(frame[y1:y2, x1:x2])
                mlp_history.append(probs)
                avg = np.mean(list(mlp_history), axis=0)
                letter = cnn_classes[MODE][int(avg.argmax())]
                conf   = float(avg.max())
                source = 'CNN'
        else:
            mlp_history.clear()
            letter, conf, source = '', 0.0, ''

    elif MODE in cnn_models:
        # CNN bez MediaPipe – centralny ROI
        roi_sz = 400
        rx1 = (fw - roi_sz) // 2; ry1 = (fh - roi_sz) // 2
        rx2 = rx1 + roi_sz;       ry2 = ry1 + roi_sz
        cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 220, 0), 2)
        probs = cnn_predict(frame[ry1:ry2, rx1:rx2])
        mlp_history.append(probs)
        avg    = np.mean(list(mlp_history), axis=0)
        letter = cnn_classes[MODE][int(avg.argmax())]
        conf   = float(avg.max())
        source = 'CNN'

    # Nie pokazuj litery podczas aktywnego gestu – czekaj na koniec sekwencji
    if gesture_state == 'MOVING' and source not in ('GRU', 'HIDE'):
        letter = ''
        conf   = 0.0

    # Litery dynamiczne tylko z GRU – MLP nie może ich wyświetlać
    if source == 'MLP' and letter in DYNAMIC_LETTERS:
        letter = ''
        conf   = 0.0

    # ── Stabilizator wyświetlania (blokada litery) ───────────────
    if letter and source != 'HIDE':
        if letter == lock_letter:
            # Aktualna litera dalej dominuje – resetuj kandydata
            lock_cand  = ''
            lock_count = 0
        elif letter == lock_cand and conf >= LOCK_CONF:
            lock_count += 1
            if lock_count >= LOCK_FRAMES:
                # Kandydat wygrał – przełącz blokadę
                lock_letter = lock_cand
                lock_cand   = ''
                lock_count  = 0
        elif conf >= LOCK_CONF:
            # Nowy kandydat
            lock_cand  = letter
            lock_count = 1
        # Pokaż zablokowaną literę zamiast surowego wyniku
        if lock_letter:
            letter = lock_letter
    else:
        # Brak dłoni – zeruj blokadę
        lock_letter = ''
        lock_cand   = ''
        lock_count  = 0

    # Pokaż ostatnią literę przez chwilę po schowaniu dłoni
    if not letter and hide_count > 0:
        hide_count -= 1
        letter = hide_letter
        conf   = hide_conf
        source = 'HIDE'
    elif letter and source != 'HIDE':
        hide_count = 0

    draw_ui(frame, letter, conf, source, bbox)
    cv2.imshow('Rozpoznawanie PJM', frame)

    key = cv2.waitKey(1) & 0xFF
    if key in (ord('q'), 27):
        break
    elif key == ord('l') and lm_detector:
        MODE = 'LANDMARKS'
        mlp_history.clear(); seq_buffer.clear(); trail_pts.clear()
        gesture_state = 'IDLE'; gesture_buffer.clear(); gesture_still = 0; motion_type = 'none'
        letter, conf, source = '', 0.0, ''
        print("→ Tryb: Landmark MLP+GRU")
    elif key == ord('p') and 'PJM' in cnn_models:
        MODE = 'PJM'
        mlp_history.clear(); seq_buffer.clear()
        letter, conf, source = '', 0.0, ''
        print("→ Tryb: CNN PJM")
    elif key == ord('a') and 'ASL' in cnn_models:
        MODE = 'ASL'
        mlp_history.clear(); seq_buffer.clear()
        letter, conf, source = '', 0.0, ''
        print("→ Tryb: CNN ASL")
    elif key == ord('s'):
        ts   = datetime.now().strftime('%Y%m%d_%H%M%S')
        path = os.path.join(SCREENSHOT_DIR, f'shot_{ts}.png')
        cv2.imwrite(path, frame)
        print(f"Zapisano: {path}")

cap.release()
cv2.destroyAllWindows()
if lm_detector:
    lm_detector.close()
print("Zamknięto.")
