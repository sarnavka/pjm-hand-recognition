"""
Oblicza dodatkowe cechy geometryczne palców z 21 landmarków MediaPipe.

Wejście:  pts  – numpy array (21, 3), współrzędne znormalizowane względem nadgarstka
Wyjście:  wektor 19 cech:
  [0:5]   – wskaźnik wyprostu każdego palca (dist tip/wrist ÷ dist MCP/wrist)
  [5:10]  – cosinus kąta zgięcia w stawie PIP każdego palca
  [10:14] – odległość czubka kciuka od czubków pozostałych palców
  [14:17] – odległości między sąsiednimi czubkami palców (index-middle, middle-ring, ring-pinky)
  [17:19] – składowe y i z wektora normalnego dłoni

Używane przez: extract_landmarks.py, extract_asl_landmarks.py, app.py
"""

import numpy as np

# ── Indeksy landmarków MediaPipe ──────────────────────────────
TIPS  = [4,  8, 12, 16, 20]   # czubki: kciuk, wskazujący, środkowy, serdeczny, mały
MCPS  = [2,  5,  9, 13, 17]   # podstawy (MCP / IP kciuka)

# Dla kąta PIP: prev_joint → pip_joint → next_joint
PIP_PREV = [2,  5,  9, 13, 17]
PIP_JOINT= [3,  6, 10, 14, 18]
PIP_NEXT = [4,  7, 11, 15, 19]

N_FINGER_FEATURES = 19


def compute_finger_features(pts):
    """
    pts: numpy array (21, 3) – landmarki po odjęciu nadgarstka
    Zwraca: numpy array float32 kształtu (19,)
    """
    pts = np.asarray(pts, dtype=np.float32)
    features = []

    # 1. Wskaźnik wyprostu palca (5 cech)
    #    dist(tip, wrist=0) / dist(MCP, wrist=0)
    #    1.0 = palec wyprostowany, <1 = zgięty
    for tip, mcp in zip(TIPS, MCPS):
        tip_d = np.linalg.norm(pts[tip])
        mcp_d = np.linalg.norm(pts[mcp])
        features.append(tip_d / (mcp_d + 1e-6))

    # 2. Cosinus kąta zgięcia w stawie PIP (5 cech)
    #    1.0 = palec wyprostowany, -1.0 = maksymalnie zgięty
    for prev, pip, nxt in zip(PIP_PREV, PIP_JOINT, PIP_NEXT):
        v1 = pts[pip] - pts[prev]
        v2 = pts[nxt]  - pts[pip]
        denom = np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6
        cos_a = float(np.clip(np.dot(v1, v2) / denom, -1.0, 1.0))
        features.append(cos_a)

    # 3. Odległość czubka kciuka od czubków 4 palców (4 cechy)
    #    mała wartość = chwyt szczypcowy (pinch)
    thumb_tip = pts[4]
    for ft in [8, 12, 16, 20]:
        features.append(float(np.linalg.norm(thumb_tip - pts[ft])))

    # 4. Odległości między sąsiednimi czubkami (3 cechy)
    #    pomaga rozróżnić N/M, V/U, W itp.
    for a, b in [(8, 12), (12, 16), (16, 20)]:
        features.append(float(np.linalg.norm(pts[a] - pts[b])))

    # 5. Składowe y i z wektora normalnego dłoni (2 cechy)
    #    określa orientację/pochylenie dłoni
    v1 = pts[5]    # nadgarstek→index MCP (już znormalizowany)
    v2 = pts[17]   # nadgarstek→pinky MCP
    normal = np.cross(v1, v2)
    norm_len = np.linalg.norm(normal) + 1e-6
    normal = normal / norm_len
    features.extend([float(normal[1]), float(normal[2])])

    return np.array(features, dtype=np.float32)
