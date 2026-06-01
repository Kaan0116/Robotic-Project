# Particle Filter Localization with AR Tags
**AIN451 / Robotics & Motion Planning — Final Project**  
Yüksel Kaan Bölükbaş

Gerçek zamanlı Monte Carlo (parçacık filtresi) ile robot lokalizasyonu.  
Robot, Gazebo Ignition Fortress simülasyonunda onboard kamerasıyla duvarlardaki ArUco markerları algılar ve bilinmeyen başlangıç konumundan kendini lokalize eder.

---

## Sistem Gereksinimleri

| | Versiyon |
|--|--|
| Ubuntu | 22.04 LTS |
| ROS 2 | Humble Hawksbill |
| Gazebo | Ignition Fortress (6.x) |
| Python | 3.10+ |
| OpenCV | 4.x |

---

## Kurulum

### 1 — ROS 2 Humble

```bash
sudo apt install software-properties-common curl -y

sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
     -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) \
     signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
     http://packages.ros.org/ros2/ubuntu \
     $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
     | sudo tee /etc/apt/sources.list.d/ros2.list

sudo apt update
sudo apt install ros-humble-desktop -y

# Her terminale ekle (ya da ~/.bashrc'ye yaz)
source /opt/ros/humble/setup.bash
```

### 2 — Gazebo Ignition Fortress

```bash
sudo apt install ignition-fortress -y
```

### 3 — ROS 2 / Gazebo köprüsü ve ek paketler

```bash
sudo apt install \
  ros-humble-ros-gz-bridge \
  ros-humble-cv-bridge \
  ros-humble-teleop-twist-keyboard \
  python3-colcon-common-extensions -y
```

### 4 — Python bağımlılıkları

```bash
pip3 install numpy matplotlib opencv-contrib-python
```

> `opencv-contrib-python` ArUco modülü için gerekli.

### 5 — Repoyu klonla ve derle

```bash
mkdir -p ~/pf_ws/src
cd ~/pf_ws/src
git clone https://github.com/Kaan0116/Robotic-Project.git pf_localization

cd ~/pf_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select pf_localization
source install/setup.bash
```

---

## Çalıştırma

**Terminal 1 — Simülasyon + tüm node'lar:**

```bash
cd ~/pf_ws
source install/setup.bash
ros2 launch pf_localization pf_sim.launch.py
```

Launch sırası (4 sn gecikme sonra PF node'ları başlar):
1. Gazebo Ignition Fortress — `ar_room.sdf`
2. ros_gz_bridge — `/odom`, `/cmd_vel`, `/camera/image`, `/clock`
3. aruco_detector — kameradan ArUco tespiti
4. particle_filter — SIR parçacık filtresi
5. visualizer — gerçek zamanlı matplotlib penceresi

**Terminal 2 — Robotu sürmek:**

```bash
source ~/pf_ws/install/setup.bash
ros2 run teleop_twist_keyboard teleop_twist_keyboard \
  --ros-args --remap cmd_vel:=/cmd_vel
```

| Tuş | Hareket |
|-----|---------|
| `i` | İleri |
| `,` | Geri |
| `j` | Sol dön |
| `l` | Sağ dön |
| `k` | Dur |
| `q/z` | Hız artır/azalt |

---

## Kontrol

```bash
source ~/pf_ws/install/setup.bash

# Kamera görüntüsü geliyor mu?
ros2 topic hz /camera/image        # ~9-10 Hz beklenir

# ArUco tespiti var mı?
ros2 topic echo /ar_detection       # robotu bir tag'a yönelt

# PF tahmin üretiyor mu?
ros2 topic echo /pf_estimate
```

Tag'a yönelince terminalde görünmesi gereken:
```
[aruco_detector]: Detected 1 marker(s) — dist=1.45 m, bearing=0.12 rad
```

---

## Ekran Görüntüleri

Visualizer, filtrenin yakınsama aşamalarını otomatik olarak `~/pf_ws/screenshots/` klasörüne kaydeder:

| Dosya | Kaydedilme koşulu |
|-------|-------------------|
| `initial_spread.png` | ESS > 0.80 × N (parçacıklar hâlâ yayılmış) |
| `partial_convergence.png` | 0.30×N < ESS < 0.80×N |
| `converged.png` | ESS < 0.30 × N (filtre yakınsamış) |

ESS = Effective Sample Size = 1 / Σwᵢ²

---

## Proje Yapısı

```
pf_localization/
├── pf_localization/
│   ├── constants.py              # Oda geometrisi, tag pozisyonları, boyut
│   ├── particle_filter_node.py   # SIR parçacık filtresi (predict/update/resample)
│   ├── aruco_detector_node.py    # Kamera tabanlı ArUco tespiti (solvePnP)
│   └── visualizer_node.py        # Gerçek zamanlı görselleştirme + screenshot
├── worlds/
│   ├── ar_room.sdf               # Gazebo dünyası (oda, 8 tag, robot)
│   └── materials/textures/
│       └── aruco_id0.png         # ArUco marker dokusu (512×512)
├── launch/
│   └── pf_sim.launch.py
├── config/
│   └── params.yaml
├── package.xml
└── setup.py
```

---

## Algoritma

### Motion Model (Predict)
Probabilistic Robotics (Thrun et al.) Alg. 5.4 — odometri tabanlı:

```
δ_rot1  = atan2(Δy, Δx) − θ_prev
δ_trans = √(Δx² + Δy²)
δ_rot2  = Δθ − δ_rot1

Gürültü: α = [0.10, 0.05, 0.05, 0.02]
```

### Sensor Model (Update)
Multi-hypothesis likelihood — tüm 8 tag eşit olarak değerlendirilir, en yakın tag'a kısayol **yapılmaz**:

```
p(z | x_particle) = Σᵢ₌₁⁸  p_dist(z_d | x, tagᵢ) × p_bear(z_α | x, tagᵢ)

σ_dist = 0.05 m  |  σ_bear = 0.05 rad
```

### Bayesian Yorum

| PF Adımı | Bayesian Karşılığı |
|----------|--------------------|
| Uniform başlatma | Prior `p(x₀)` |
| Predict | Geçiş `p(xₜ | xₜ₋₁, uₜ)` |
| Update | Likelihood `p(zₜ | xₜ)` |
| Ağırlıklı parçacıklar | Posterior `p(xₜ | z₁:ₜ, u₁:ₜ)` |
| Resample | Posterior'dan Monte Carlo örnekleme |

### Resampling
Low-variance resampling — `N_eff < N/2` olduğunda tetiklenir.

---

## Dünya Haritası

```
     ┌──────────────────────────────┐
     │  T2(-0.3,2.4)  T3(2.0,2.4)  │  ← Kuzey duvarı
     │                               │
T5   │                               │  T6
(-2.9│                               │(2.9
1.5) │                               │0.5)
     │                               │
T4   │                               │  T7
(-2.9│                               │(2.9
-0.5)│                               │-1.5)
     │                               │
     │  T0(-1.5,-2.4) T1(1.2,-2.4)  │  ← Güney duvarı
     └──────────────────────────────┘
```

Oda: 6 m × 5 m × 3 m | Tag boyutu: 0.25 m × 0.25 m | Tüm tag'lar ID=0

---

## Parametreler

| Parametre | Değer | Açıklama |
|-----------|-------|----------|
| `N_PARTICLES` | 2000 | Parçacık sayısı |
| `SIGMA_DIST` | 0.05 m | Sensor modeli mesafe std |
| `SIGMA_BEAR` | 0.05 rad | Sensor modeli açı std |
| `ALPHA` | [0.10, 0.05, 0.05, 0.02] | Motion model gürültü katsayıları |

---

## Sorun Giderme

**`ros_gz_bridge` bulunamıyor:**
```bash
sudo apt install ros-humble-ros-gz-bridge
# Eski sistemlerde:
sudo apt install ros-humble-ros-ign-bridge
# ve pf_sim.launch.py içinde package='ros_ign_bridge' yap
```

**Gazebo açılmıyor:**
```bash
ign gazebo --force-version 6
```

**Matplotlib penceresi çıkmıyor:**
```bash
sudo apt install python3-tk
# veya visualizer_node.py'de: matplotlib.use('Qt5Agg')
```

**`cv_bridge` import hatası:**
```bash
sudo apt install ros-humble-cv-bridge
```

---

## Lisans

MIT
