"""
Nagrywanie danych do bazy PJM – statyczne zdjęcia i dynamiczne sekwencje.

Tryb STATYCZNY:
  Zapisuje zdjęcia do data/pjm_dataset/<litera>/
  Po sesji automatycznie aktualizuje data/pjm_landmarks.npz

Tryb DYNAMICZNY:
  Zapisuje sekwencje (30 klatek × 65) do data/pjm_sequences/<litera>/

Sterowanie (w każdym trybie):
  SPACJA  – zrób zdjęcie / zacznij nagrywanie sekwencji
  R       – usuń ostatnią próbkę
  N       – pomiń do następnej litery
  B       – wróć do menu
  Q / Esc – wyjdź
"""

import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MEDIAPIPE_DISABLE_GPU'] = '1'

import sys
import argparse
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

sys.path.insert(0, os.path.dirname(__file__))
from finger_features import compute_finger_features

# ── PIL – obsługa polskich znaków ────────────────────────────
try:
    from PIL import ImageFont, ImageDraw, Image as PILImage
    _PIL_OK = True
except ImportError:
    _PIL_OK = False

_font_cache: dict = {}

def _get_font(size: int):
    if size in _font_cache:
        return _font_cache[size]
    candidates = [
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibri.ttf",
        r"C:\Windows\Fonts\segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    ]
    font = None
    for path in candidates:
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass
    if font is None:
        font = ImageFont.load_default()
    _font_cache[size] = font
    return font


def put_text_pl(frame, text, pos, font_size, color_bgr, stroke=0):
    """Rysuje tekst Unicode (polskie znaki) na klatce OpenCV. pos = lewy-górny róg."""
    if not _PIL_OK:
        cv2.putText(frame, text, (pos[0], pos[1] + font_size),
                    cv2.FONT_HERSHEY_SIMPLEX, font_size / 28, color_bgr, 1)
        return
    font = _get_font(font_size)
    img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    kw = {'stroke_width': stroke, 'stroke_fill': (0, 0, 0)} if stroke > 0 else {}
    draw.text(pos, text, font=font, fill=rgb, **kw)
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def draw_big_letter(frame, letter, color_bgr):
    """Rysuje dużą wyśrodkowaną literę na klatce (obsługuje polskie znaki)."""
    h, w = frame.shape[:2]
    if not _PIL_OK:
        fs = 6
        ts = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, fs, 10)[0]
        cx = (w - ts[0]) // 2
        cy = h // 2 + ts[1] // 2
        cv2.putText(frame, letter, (cx + 3, cy + 3),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), 14)
        cv2.putText(frame, letter, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, color_bgr, 10)
        return
    font_size = min(h // 3, 200)
    font = _get_font(font_size)
    img_pil = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    bbox = font.getbbox(letter)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]
    cx = (w - tw) // 2 - bbox[0]
    cy = (h - th) // 2 - bbox[1]
    draw.text((cx + 3, cy + 3), letter, font=font, fill=(0, 0, 0),
              stroke_width=4, stroke_fill=(0, 0, 0))
    rgb = (color_bgr[2], color_bgr[1], color_bgr[0])
    draw.text((cx, cy), letter, font=font, fill=rgb,
              stroke_width=2, stroke_fill=(0, 0, 0))
    frame[:] = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# ── KONFIGURACJA ──────────────────────────────────────────────
DATASET_DIR  = 'data/pjm_dataset'
SEQ_DIR      = 'data/pjm_sequences'
LANDMARKS_NPZ = 'data/pjm_landmarks.npz'
TASK_PATH    = 'models/hand_landmarker.task'

SEQ_LEN  = 30   # klatek w sekwencji (~1 s przy 30 fps)
N_TARGET = 120  # docelowa liczba próbek na klasę

# Litery wymagające dłuższego nagrania
SEQ_LEN_OVERRIDE = {
    'Ź': 40,
    'Ż': 50,
}

# Wszystkie litery PJM daktylografii
ALL_LETTERS = [
    'A', 'Ą', 'B', 'C', 'Ć', 'CH', 'CZ', 'D', 'E', 'Ę',
    'F', 'G', 'H', 'I', 'J', 'K', 'L', 'Ł', 'M', 'N',
    'Ń', 'O', 'Ó', 'P', 'R', 'RZ', 'S', 'Ś', 'SZ', 'T',
    'U', 'W', 'Y', 'Z', 'Ź', 'Ż',
]

# Znaki wymagające ruchu (sekwencje) – reszta statyczna
DYNAMIC_LETTERS = {
    'Ą', 'Ć', 'CH', 'CZ', 'D', 'Ę', 'F', 'G', 'H',
    'J', 'K', 'Ł', 'Ń', 'Ó', 'RZ', 'Ś', 'SZ', 'Z', 'Ź', 'Ż',
}

os.makedirs(DATASET_DIR, exist_ok=True)
os.makedirs(SEQ_DIR, exist_ok=True)

# ── MEDIAPIPE ─────────────────────────────────────────────────
base_opts = mp_python.BaseOptions(model_asset_path=TASK_PATH)
opts = mp_vision.HandLandmarkerOptions(
    base_options=base_opts, num_hands=1,
    min_hand_detection_confidence=0.4,
    min_hand_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    running_mode=mp_vision.RunningMode.IMAGE,
)
detector = mp_vision.HandLandmarker.create_from_options(opts)


# ── POMOCNICZE ────────────────────────────────────────────────
def detect_hand(frame_bgr):
    """Zwraca (lm_list, lm_vec_82, wrist_xy) lub (None, None, None)."""
    rgb    = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    if not result.hand_landmarks:
        return None, None, None
    lm    = result.hand_landmarks[0]
    wrist = lm[0]
    pts   = np.array([[p.x - wrist.x, p.y - wrist.y, p.z - wrist.z]
                       for p in lm], dtype=np.float32)
    vec63  = pts.flatten()
    vec82  = np.concatenate([vec63, compute_finger_features(pts)])
    return lm, vec82, (wrist.x, wrist.y)


def draw_skeleton(frame, lm_list):
    if lm_list is None:
        return
    h, w = frame.shape[:2]
    conns = [(0,1),(1,2),(2,3),(3,4),
             (0,5),(5,6),(6,7),(7,8),
             (0,9),(9,10),(10,11),(11,12),
             (0,13),(13,14),(14,15),(15,16),
             (0,17),(17,18),(18,19),(19,20),
             (5,9),(9,13),(13,17)]
    for a, b in conns:
        ax = int(lm_list[a].x * w); ay = int(lm_list[a].y * h)
        bx = int(lm_list[b].x * w); by = int(lm_list[b].y * h)
        cv2.line(frame, (ax, ay), (bx, by), (180, 60, 220), 2)
    for p in lm_list:
        cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (220, 120, 255), -1)


def count_images(letter):
    d = os.path.join(DATASET_DIR, letter)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))])


def count_seqs(letter):
    d = os.path.join(SEQ_DIR, letter)
    if not os.path.exists(d):
        return 0
    return len([f for f in os.listdir(d) if f.endswith('.npy')])


def next_img_idx(letter):
    d = os.path.join(DATASET_DIR, letter)
    if not os.path.exists(d):
        return 1
    existing = [f for f in os.listdir(d)
                if f.lower().endswith(('.jpg', '.png'))]
    nums = []
    for f in existing:
        name = os.path.splitext(f)[0]
        digits = ''.join(c for c in name if c.isdigit())
        if digits:
            nums.append(int(digits))
    return max(nums, default=0) + 1


def next_seq_idx(letter):
    d = os.path.join(SEQ_DIR, letter)
    if not os.path.exists(d):
        return 0
    files = [f for f in os.listdir(d) if f.endswith('.npy')]
    if not files:
        return 0
    nums = [int(f.replace('seq_', '').replace('.npy', ''))
            for f in files if f.startswith('seq_')]
    return max(nums, default=-1) + 1


def save_image(letter, idx, frame):
    d = os.path.join(DATASET_DIR, letter)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'{letter}{idx}.jpg')
    cv2.imwrite(path, frame)
    return path


def save_seq(letter, idx, buffer):
    d = os.path.join(SEQ_DIR, letter)
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f'seq_{idx:04d}.npy')
    np.save(path, np.array(buffer, dtype=np.float32))
    return path


def delete_last_image(letter, idx):
    path = os.path.join(DATASET_DIR, letter, f'{letter}{idx}.jpg')
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


def delete_last_seq(letter, idx):
    path = os.path.join(SEQ_DIR, letter, f'seq_{idx:04d}.npy')
    if os.path.exists(path):
        os.remove(path)
        return True
    return False


# ── AKTUALIZACJA LANDMARKÓW ───────────────────────────────────
def update_landmarks():
    """Przebudowuje pjm_landmarks.npz ze wszystkich zdjęć w pjm_dataset."""
    print("\n  Aktualizuję pjm_landmarks.npz...")
    classes = sorted([d for d in os.listdir(DATASET_DIR)
                      if os.path.isdir(os.path.join(DATASET_DIR, d))])
    class_idx = {c: i for i, c in enumerate(classes)}
    X, y, skipped = [], [], 0

    for cls in classes:
        cls_dir = os.path.join(DATASET_DIR, cls)
        files   = [f for f in os.listdir(cls_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        detected = 0
        for fname in files:
            img = cv2.imread(os.path.join(cls_dir, fname))
            if img is None:
                skipped += 1
                continue
            _, vec82, _ = detect_hand(img)
            if vec82 is not None:
                X.append(vec82)
                y.append(class_idx[cls])
                detected += 1
            else:
                skipped += 1
        print(f"    {cls:<4}: {detected}/{len(files)}")

    if X:
        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y, dtype=np.int32)
        np.savez(LANDMARKS_NPZ, X=X_arr, y=y_arr, classes=np.array(classes))
        print(f"  Zapisano {LANDMARKS_NPZ}  ({len(X_arr)} próbek, {len(classes)} klas)")
    else:
        print("  Brak danych do zapisu!")


# ── EKRAN GŁÓWNY ──────────────────────────────────────────────
def draw_menu(cap):
    """Rysuje menu na ekranie kamery. Zwraca key."""
    ret, frame = cap.read()
    if not ret:
        return frame, -1
    frame = cv2.flip(frame, 1)
    h, w  = frame.shape[:2]

    # Przyciemnione tło
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # Tytuł
    cv2.putText(frame, "NAGRYWANIE BAZY PJM",
                (w//2 - 220, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 80, 255), 3)
    cv2.putText(frame, "Wybierz tryb:",
                (w//2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    # Opcje
    options = [
        ("S", "Nagraj STATYCZNE zdjecia (brakujace / malo probek)", (80, 200, 80)),
        ("D", "Nagraj DYNAMICZNE sekwencje (znaki z ruchem)",       (80, 160, 255)),
        ("I", "Pokaz statystyki bazy danych",                        (200, 200, 80)),
        ("U", "Zaktualizuj pjm_landmarks.npz",                      (200, 120, 80)),
        ("Q", "Wyjdz",                                               (100, 100, 100)),
    ]
    for i, (key, desc, color) in enumerate(options):
        y_pos = 150 + i * 50
        cv2.putText(frame, f"[{key}]", (w//2 - 200, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
        cv2.putText(frame, desc, (w//2 - 140, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1)

    cv2.imshow('Nagrywanie PJM', frame)
    key = cv2.waitKey(30) & 0xFF
    return frame, key


# ── EKRAN STATYSTYK ───────────────────────────────────────────
def show_stats(cap):
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, "STATYSTYKI BAZY PJM",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 80, 255), 2)
        cv2.putText(frame, "Litera | Zdjecia | Sekwencje | Status",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.line(frame, (20, 80), (w - 20, 80), (80, 80, 80), 1)

        cols = 2
        col_w = (w - 40) // cols
        for i, letter in enumerate(ALL_LETTERS):
            col   = i % cols
            row   = i // cols
            x_off = 20 + col * col_w
            y_off = 100 + row * 22

            n_img = count_images(letter)
            n_seq = count_seqs(letter)
            is_dyn = letter in DYNAMIC_LETTERS

            if is_dyn:
                ok = n_seq >= N_TARGET
                status = f"seq:{n_seq}"
            else:
                ok = n_img >= N_TARGET
                status = f"img:{n_img}"

            color = (80, 220, 80) if ok else (80, 80, 220) if (n_img > 0 or n_seq > 0) else (80, 80, 80)
            marker = "D" if is_dyn else "S"
            put_text_pl(frame, f"[{marker}] {letter:<3} {status}",
                        (x_off, y_off - 14), 14, color)

        cv2.putText(frame, "B / Esc = powrot do menu",
                    (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (150, 150, 150), 1)
        cv2.imshow('Nagrywanie PJM', frame)
        k = cv2.waitKey(30) & 0xFF
        if k in (ord('b'), ord('q'), 27):
            break


# ── WYBÓR LITERY ──────────────────────────────────────────────
def choose_letter(cap, mode):
    """Wyświetla listę liter. Zwraca wybraną lub None."""
    candidates = [l for l in ALL_LETTERS
                  if (mode == 'dynamic') == (l in DYNAMIC_LETTERS)]
    selected = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        title = "WYBIERZ LITERE (DYNAMICZNA)" if mode == 'dynamic' else "WYBIERZ LITERE (STATYCZNA)"
        cv2.putText(frame, title,
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (200, 80, 255), 2)
        cv2.putText(frame, "↑↓ / numery rzad + kolumna | Enter=wybierz | B=menu | A=wszystkie",
                    (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        cols = 4
        col_w = (w - 40) // cols
        for i, letter in enumerate(candidates):
            col   = i % cols
            row   = i // cols
            x_off = 20 + col * col_w
            y_off = 100 + row * 35

            n = count_seqs(letter) if mode == 'dynamic' else count_images(letter)
            bar_len = min(int(n / N_TARGET * 60), 60)
            bar = '█' * bar_len + '░' * (60 - bar_len)

            bg_color = (60, 30, 80) if i == selected else (20, 20, 20)
            cv2.rectangle(frame,
                          (x_off - 4, y_off - 18),
                          (x_off + col_w - 12, y_off + 10),
                          bg_color, -1)

            warn = " !" if n < 60 else ""
            txt_color = (80, 220, 80) if n >= N_TARGET else (220, 200, 80) if n >= 60 else (220, 80, 80)
            put_text_pl(frame, f"{letter:<4} {n:>3}{warn}",
                        (x_off, y_off - 16), 16, txt_color)

        put_text_pl(frame, f"Wybrany: {candidates[selected]}",
                    (20, h - 40), 22, (200, 80, 255))
        cv2.imshow('Nagrywanie PJM', frame)
        k = cv2.waitKey(30) & 0xFF

        if k == 82 or k == ord('w'):    # strzałka w górę / W
            selected = (selected - cols) % len(candidates)
        elif k == 84 or k == ord('s'):  # strzałka w dół / S
            selected = (selected + cols) % len(candidates)
        elif k == 81 or k == ord('a'):  # strzałka w lewo / A
            selected = (selected - 1) % len(candidates)
        elif k == 83 or k == ord('d'):  # strzałka w prawo / D
            selected = (selected + 1) % len(candidates)
        elif k == 13:  # Enter
            return candidates[selected]
        elif k in (ord('b'), 27):
            return None


# ── NAGRYWANIE STATYCZNE ──────────────────────────────────────
def record_static(cap, letter):
    """Nagrywa zdjęcia statyczne dla danej litery."""
    img_idx = next_img_idx(letter)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        lm_list, vec82, _ = detect_hand(frame)
        hand_ok = lm_list is not None

        if hand_ok:
            draw_skeleton(frame, lm_list)

        # Panel górny
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 10, 30), -1)
        put_text_pl(frame, f"STATYCZNE  [{letter}]",
                    (10, 12), 30, (200, 80, 255))
        n_cur = count_images(letter)
        cv2.putText(frame, f"Zapisanych: {n_cur}  cel: {N_TARGET}  |  B=menu  R=usun  Q=wyjdz",
                    (w - 510, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        # Pasek postępu
        prog = min(n_cur / N_TARGET, 1.0)
        bw   = w - 40
        cv2.rectangle(frame, (20, 68), (20 + bw, 80), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 68), (20 + int(bw * prog), 80), (180, 60, 220), -1)

        # Duża litera
        draw_big_letter(frame, letter, (160, 50, 200))

        # Dolny status
        if hand_ok:
            cv2.putText(frame, "Dlon wykryta – SPACJA = zapisz zdjecie",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
        else:
            cv2.putText(frame, "Pokaz dlon do kamery",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 1)

        cv2.imshow('Nagrywanie PJM', frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27):
            return 'quit'
        elif k == ord('b'):
            return 'menu'
        elif k == ord('n'):
            return 'next'
        elif k == ord('r'):
            if img_idx > 1:
                if delete_last_image(letter, img_idx - 1):
                    img_idx -= 1
                    print(f"  Usunięto {letter}{img_idx}.jpg")
        elif k == ord(' '):
            if hand_ok:
                path = save_image(letter, img_idx, frame)
                print(f"  Zapisano {path}")
                img_idx += 1
                # Błysk
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (w, h), (255, 255, 255), 20)
                cv2.imshow('Nagrywanie PJM', flash)
                cv2.waitKey(80)

    return 'quit'


# ── NAGRYWANIE DYNAMICZNE ─────────────────────────────────────
def record_dynamic(cap, letter):
    """Nagrywa sekwencje dynamiczne dla danej litery."""
    cur_seq_len = SEQ_LEN_OVERRIDE.get(letter, SEQ_LEN)
    seq_idx = next_seq_idx(letter)
    state       = 'IDLE'    # IDLE | COUNTDOWN | RECORDING | FLASH
    cdown_t     = 0
    buffer      = []
    flash_t     = 0
    prev_wrist  = None      # poprzednia bezwzgl. pozycja nadgarstka

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        lm_list, vec82, wrist_xy = detect_hand(frame)
        hand_ok = lm_list is not None

        if hand_ok:
            draw_skeleton(frame, lm_list)
            vec63 = vec82[:63]
            # Prędkość nadgarstka (delta bezwzgl. pozycji)
            if prev_wrist is not None:
                dw = np.array([wrist_xy[0] - prev_wrist[0],
                               wrist_xy[1] - prev_wrist[1]], dtype=np.float32)
            else:
                dw = np.zeros(2, dtype=np.float32)
            prev_wrist = wrist_xy
            lm_vec65 = np.concatenate([vec63, dw])
        else:
            lm_vec65   = None
            prev_wrist = None

        # Panel górny
        cv2.rectangle(frame, (0, 0), (w, 65), (10, 20, 30), -1)
        put_text_pl(frame, f"DYNAMICZNE  [{letter}]",
                    (10, 12), 30, (80, 160, 255))
        n_cur = count_seqs(letter)
        cv2.putText(frame, f"Sekwencji: {n_cur}  cel: {N_TARGET}  |  B=menu  R=usun  Q=wyjdz",
                    (w - 510, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 150), 1)

        # Pasek postępu
        prog = min(n_cur / N_TARGET, 1.0)
        bw   = w - 40
        cv2.rectangle(frame, (20, 68), (20 + bw, 80), (40, 40, 40), -1)
        cv2.rectangle(frame, (20, 68), (20 + int(bw * prog), 80), (60, 140, 220), -1)

        # Duża litera
        draw_big_letter(frame, letter, (40, 120, 200))

        # ── Maszyna stanów ────────────────────────────────────
        if state == 'IDLE':
            if hand_ok:
                cv2.putText(frame, "Dlon OK – SPACJA = zacznij nagrywac sekwencje",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 220, 80), 2)
            else:
                cv2.putText(frame, "Pokaz dlon do kamery",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (80, 80, 200), 1)

        elif state == 'COUNTDOWN':
            elapsed   = (cv2.getTickCount() - cdown_t) / cv2.getTickFrequency()
            secs_left = 3 - int(elapsed)
            if secs_left <= 0:
                if hand_ok:
                    state  = 'RECORDING'
                    buffer = []
                else:
                    state = 'IDLE'
            else:
                txt = str(secs_left)
                ts2 = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 5, 10)[0]
                cv2.putText(frame, txt,
                            ((w - ts2[0]) // 2, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 165, 255), 10)

        elif state == 'RECORDING':
            if lm_vec65 is not None:
                buffer.append(lm_vec65)
            else:
                state      = 'IDLE'
                buffer     = []
                prev_wrist = None
                cv2.putText(frame, "Reka znikla – sprobuj ponownie",
                            (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 80, 220), 2)

            if state == 'RECORDING':
                prog2 = len(buffer) / cur_seq_len
                cv2.rectangle(frame, (20, h - 45), (20 + bw, h - 25), (40, 40, 40), -1)
                cv2.rectangle(frame, (20, h - 45), (20 + int(bw * prog2), h - 25), (0, 80, 220), -1)
                # Migająca czerwona kropka
                if int(cv2.getTickCount() / cv2.getTickFrequency() * 3) % 2 == 0:
                    cv2.circle(frame, (w - 30, 30), 12, (0, 0, 220), -1)
                cv2.putText(frame, f"NAGRYWA  {len(buffer)}/{cur_seq_len}",
                            (10, h - 52), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80, 140, 255), 2)

                if len(buffer) >= cur_seq_len:
                    path = save_seq(letter, seq_idx, buffer)
                    print(f"  Zapisano {path}")
                    seq_idx += 1
                    state   = 'FLASH'
                    flash_t = cv2.getTickCount()

        elif state == 'FLASH':
            cv2.rectangle(frame, (0, 0), (w, h), (0, 60, 0), 10)
            cv2.putText(frame, f"Zapisano! ({seq_idx}/{N_TARGET})",
                        (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (80, 255, 80), 2)
            elapsed = (cv2.getTickCount() - flash_t) / cv2.getTickFrequency()
            if elapsed > 0.5:
                state = 'IDLE'

        cv2.imshow('Nagrywanie PJM', frame)
        k = cv2.waitKey(1) & 0xFF

        if k in (ord('q'), 27):
            return 'quit'
        elif k == ord('b'):
            return 'menu'
        elif k == ord('n'):
            return 'next'
        elif k == ord('r'):
            if seq_idx > 0:
                if delete_last_seq(letter, seq_idx - 1):
                    seq_idx -= 1
                    state   = 'IDLE'
                    print(f"  Usunięto sekwencję {seq_idx}")
        elif k == ord(' '):
            if state == 'IDLE' and hand_ok:
                state   = 'COUNTDOWN'
                cdown_t = cv2.getTickCount()

    return 'quit'


# ── WYBÓR KAMERY ──────────────────────────────────────────────
# Wymuszone na DirectShow na Windows - wirtualne kamery typu Iriun/DroidCam
# często rejestrują się TYLKO jako filtr DirectShow, a nie Media Foundation
# (domyślny backend OpenCV od pewnej wersji). Bez tego indeksy wykryte przy
# probingu mogłyby nie odpowiadać indeksom otwieranym później.
CAM_BACKEND = cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY


def probe_camera(idx, attempts=5):
    """Sprawdza czy kamera o danym indeksie faktycznie dostarcza obraz."""
    cap = cv2.VideoCapture(idx, CAM_BACKEND)
    if not cap.isOpened():
        cap.release()
        return False
    ok = False
    for _ in range(attempts):
        ret, frame = cap.read()
        if ret and frame is not None:
            ok = True
            break
    cap.release()
    return ok


def find_cameras(max_idx=6):
    """Zwraca listę indeksów kamer, które faktycznie dają obraz
    (np. wbudowana kamera + wirtualna z Iriun/DroidCam)."""
    return [i for i in range(max_idx) if probe_camera(i)]


VIRTUAL_CAM_HINTS = ('iriun', 'droidcam', 'epoccam', 'camo', 'ivcam')


def get_camera_names():
    """Windows: zwraca nazwy urządzeń wideo (DirectShow) w kolejności
    indeksów OpenCV, albo None gdy niedostępne (brak pygrabber / nie-Windows)."""
    if os.name != 'nt':
        return None
    try:
        from pygrabber.dshow_graph import FilterGraph
        return FilterGraph().get_input_devices()
    except Exception:
        return None


def describe_camera(idx, names):
    name = names[idx] if names and idx < len(names) else f"kamera {idx}"
    hint = "   ← to prawdopodobnie Iriun/DroidCam" \
        if any(h in name.lower() for h in VIRTUAL_CAM_HINTS) else ""
    return f"[{idx}] {name}{hint}"


def select_camera_visually(found, names):
    """Podgląd na żywo z każdej znalezionej kamery, z podpisem –
    przełączanie N, wybór ENTER/SPACJA, anulowanie Q (bierze pierwszą)."""
    pos      = 0
    cur_idx  = None
    cap      = None
    try:
        while True:
            idx = found[pos]
            if idx != cur_idx:
                if cap is not None:
                    cap.release()
                cap     = cv2.VideoCapture(idx, CAM_BACKEND)
                cur_idx = idx

            ret, frame = cap.read()
            if not ret or frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, 70), (10, 10, 10), -1)
            cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
            label = describe_camera(idx, names)
            put_text_pl(frame, f"{pos + 1}/{len(found)}:  {label}",
                        (15, 14), 24, (80, 220, 80))
            cv2.putText(frame,
                        "N = nastepna kamera  |  ENTER/SPACJA = wybierz  |  Q = anuluj",
                        (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

            cv2.imshow('Wybor kamery – podglad na zywo', frame)
            k = cv2.waitKey(30) & 0xFF
            if k in (13, 32):          # Enter / Spacja
                return idx
            elif k == ord('n'):
                pos = (pos + 1) % len(found)
            elif k in (ord('q'), 27):  # Q / Esc
                return found[0]
    finally:
        if cap is not None:
            cap.release()
        cv2.destroyWindow('Wybor kamery – podglad na zywo')


def open_camera(forced_idx=None):
    if forced_idx is not None:
        cap = cv2.VideoCapture(forced_idx, CAM_BACKEND)
        if not cap.isOpened():
            print(f"  Nie udało się otworzyć kamery o indeksie {forced_idx}.")
            sys.exit(1)
        return cap

    print("  Szukam dostępnych kamer...")
    found = find_cameras()
    names = get_camera_names()

    if not found:
        print("  Nie znaleziono żadnej działającej kamery.")
        print("  Jeśli używasz Iriun/DroidCam – upewnij się, że aplikacja")
        print("  na telefonie jest otwarta i połączona z komputerem, a potem")
        print("  uruchom skrypt ponownie.")
        sys.exit(1)

    if len(found) == 1:
        idx = found[0]
        print(f"  Znaleziono kamerę: {describe_camera(idx, names)} – używam jej.")
    else:
        print(f"  Znaleziono {len(found)} kamery:")
        for idx in found:
            print(f"    {describe_camera(idx, names)}")
        print("  Otwieram podgląd na żywo – klawisz N przełącza kamerę, "
              "ENTER/SPACJA wybiera.")
        idx = select_camera_visually(found, names)
        print(f"  Wybrano: {describe_camera(idx, names)}")

    return cv2.VideoCapture(idx, CAM_BACKEND)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=None,
                         help='Indeks kamery – pomija automatyczne wykrywanie')
    return parser.parse_args()


# ── GŁÓWNA PĘTLA ──────────────────────────────────────────────
def main():
    args = parse_args()
    cap = open_camera(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\n" + "=" * 60)
    print("  NAGRYWANIE BAZY PJM")
    print("=" * 60)

    # Pokaż braki przy starcie
    missing_dynamic = [l for l in DYNAMIC_LETTERS if count_seqs(l) == 0]
    low_static      = [l for l in ALL_LETTERS
                       if l not in DYNAMIC_LETTERS and count_images(l) < 60]
    if missing_dynamic:
        print(f"  Brak sekwencji dla: {missing_dynamic}")
    if low_static:
        print(f"  Mało zdjęć (<60) dla: {low_static}")
    print()

    while True:
        _, key = draw_menu(cap)

        if key in (ord('q'), 27):
            break

        elif key == ord('s'):
            letter = choose_letter(cap, 'static')
            if letter is None:
                continue
            result = record_static(cap, letter)
            if result == 'quit':
                break

        elif key == ord('d'):
            letter = choose_letter(cap, 'dynamic')
            if letter is None:
                continue
            result = record_dynamic(cap, letter)
            if result == 'quit':
                break

        elif key == ord('i'):
            show_stats(cap)

        elif key == ord('u'):
            cv2.destroyAllWindows()
            update_landmarks()
            print("  Gotowe. Nacisnij Enter aby kontynuowac...")
            input()

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

    # Podsumowanie końcowe
    print("\n" + "=" * 60)
    print("  PODSUMOWANIE SESJI")
    print("=" * 60)
    for letter in ALL_LETTERS:
        is_dyn = letter in DYNAMIC_LETTERS
        if is_dyn:
            n   = count_seqs(letter)
            typ = "seq"
        else:
            n   = count_images(letter)
            typ = "img"
        bar = '█' * min(n * 20 // N_TARGET, 20) + '░' * max(0, 20 - n * 20 // N_TARGET)
        ok  = "✓" if n >= N_TARGET else "!"
        print(f"  {ok} {letter:<4} [{typ}] {bar} {n:>4}/{N_TARGET}")
    print("=" * 60)
    print("  Aby przetrenować model:")
    print("  1. python extract_landmarks.py   (statyczne)")
    print("  2. python train_landmark_model.py")
    print("  3. python train_lstm.py          (dynamiczne)")
    print("=" * 60)


if __name__ == '__main__':
    main()
