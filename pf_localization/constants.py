"""
Shared constants: tag map, room geometry.
Oda: 6m × 5m  (x: -3.0..+3.0, y: -2.5..+2.5)
8 AR tag — hepsi aynı ID (0), asimetrik yerleşim.
"""
import numpy as np

# 2-D tag positions in world frame [m]
TAG_POSITIONS = np.array([
    (-1.50, -2.46),   # tag 0  S1
    ( 1.20, -2.46),   # tag 1  S2
    (-0.30,  2.46),   # tag 2  N1
    ( 2.00,  2.46),   # tag 3  N2
    (-2.96, -0.50),   # tag 4  W1
    (-2.96,  1.50),   # tag 5  W2
    ( 2.96,  0.50),   # tag 6  E1
    ( 2.96, -1.50),   # tag 7  E2
], dtype=float)

TAG_Z    = 1.00   # tag center height [m]
TAG_SIZE = 0.25   # physical marker side length [m]

ROOM_X_MIN, ROOM_X_MAX = -3.0, 3.0
ROOM_Y_MIN, ROOM_Y_MAX = -2.5, 2.5

N_TAGS = len(TAG_POSITIONS)
