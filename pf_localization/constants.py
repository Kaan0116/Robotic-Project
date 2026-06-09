"""
Shared constants: tag map, room geometry.
Oda: 6m × 5m  (x: -3.0..+3.0, y: -2.5..+2.5)
8 AR tag — hepsi aynı ID (0), asimetrik yerleşim.
"""
import numpy as np

# 2-D tag positions in world frame [m]
TAG_POSITIONS = np.array([
    (-2.10, -2.46),   # tag 0  S1 – South wall,
    ( 0.80, -2.46),   # tag 1  S2 – South wall, 
    (-0.70,  2.46),   # tag 2  N1 – North wall,
    ( 2.30,  2.46),   # tag 3  N2 – North wall,
    (-2.96, -1.80),   # tag 4  W1 – West wall,
    (-2.96,  1.10),   # tag 5  W2 – West wall, 
    ( 2.96,  0.20),   # tag 6  E1 – East wall, 
    ( 2.96, -0.50),   # tag 7  E2 – East wall,
], dtype=float)

TAG_Z    = 1.00   # tag center height [m]
TAG_SIZE = 0.25   # physical marker side length [m]

ROOM_X_MIN, ROOM_X_MAX = -3.0, 3.0
ROOM_Y_MIN, ROOM_Y_MAX = -2.5, 2.5

N_TAGS = len(TAG_POSITIONS)
