# Football_Player_Tracking_System
# ⚽ Advanced Football Player Tracking System
> **Logic over Model**: Heavy한 Re-ID 모델 없이, Pandas 기반의 후처리 알고리즘으로 끊김 없는 트래킹 구현

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv10](https://img.shields.io/badge/YOLO-v10m-green)
![Pandas](https://img.shields.io/badge/Pandas-Data_Processing-orange)


```mermaid
graph TD
    %% 스타일 정의
    classDef ai fill:#e1f5fe,stroke:#01579b,stroke-width:2px,color:#000;
    classDef logic fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,stroke-dasharray: 5 5,color:#000;
    classDef data fill:#e0f2f1,stroke:#00695c,stroke-width:2px,color:#000;

    %% 노드 정의
    Start((Start)) --> Input[/"🎥 Input Video &<br/>Target Coordinates"/]
    
    subgraph Phase1 [Phase 1: Enhanced Inference]
        direction TB
        Upscale["🔍 Upscaling (x1.5)<br/>Small Object Enhancement"]:::ai
        Detect["🤖 YOLOv10 Inference<br/>Object Detection"]:::ai
        Track["ID BoT-SORT Tracker<br/>Target Locking"]:::ai
        
        Input --> Upscale --> Detect --> Track
    end

    Track --> RawData[("📂 Raw Track Data<br/>(Fragmented IDs)")]:::data

    subgraph Phase2 [Phase 2: Data Engineering Logic]
        direction TB
        Stitch["🔗 Track Stitching<br/>(Spatio-temporal Distance Matching)"]:::logic
        Interp["📈 Linear Interpolation<br/>(Fill Missing Frames)"]:::logic
        Smooth["Correction<br/>(Noise Reduction)"]:::logic
        
        RawData --> Stitch --> Interp --> Smooth
    end

    subgraph Phase3 [Phase 3: Tactical Analysis]
        direction TB
        Map["CONST Linear Mapping<br/>(Pixel → Meter)"]
        Zone["grid 18-Zone Calculation<br/>(Tactical Heatmap)"]
        
        Smooth --> Map --> Zone
    end

    Zone --> Output[/"📊 Final Report &<br/>Tracking Video"/]:::data

    %% 관계선 설명
    linkStyle default stroke-width:2px,fill:none,stroke:#333;
```
```

## 📌 Project Overview
축구 중계 영상(Broadcast View)에서 특정 선수를 추적하여 **18-Zone 점유율 및 히트맵**을 분석하는 파이프라인입니다.
선수 교차(Occlusion)와 작은 객체 크기로 인한 Detection 실패 문제를 **Post-processing**으로 해결했습니다.

## 💥 The Challenge
1.  **Frequent ID Switching**: 선수들이 겹치거나 교차할 때 Tracker ID가 바뀌어 추적 연속성이 깨짐.
2.  **Small Object Failure**: 원거리 앵글(Long shot) 특성상 선수 객체가 너무 작아 YOLO가 놓치는 경우 빈번 발생.
3.  **Resource Constraints**: 실시간 분석을 위해 무거운 Re-ID(DeepSORT 등) 모델을 추가 학습시키기 부담스러움.

## 💡 Solution: "Logic over Model"
딥러닝 모델의 성능에만 의존하지 않고, **알고리즘적 후처리**를 통해 성능을 극대화했습니다.

### 1. Spatio-temporal Track Stitching (트랙 스티칭)
끊어진 트랙 ID들을 연결하기 위해 **Spatio-temporal distance**를 계산하여 동일 인물일 가능성이 높은 ID를 강제로 병합합니다.
* `Candidate Search`: 타겟 ID가 사라진 시점(`last_frame`) 이후 `N` 프레임 내에 새로 등장한 ID 탐색.
* `Distance Matching`: 마지막 위치와 후보 ID의 시작 위치 간 유클리드 거리가 임계값(`150px`) 이내인 경우 병합.

### 2. Upscaling Inference Strategy
작은 객체 탐지율을 높이기 위해, 추론 단계에서 입력 이미지를 **1.5배 Upscaling**하여 주입합니다.
* **Result**: 뭉개져서 잡히지 않던 선수의 윤곽선이 뚜렷해지며 Detection Recall 대폭 상승.

### 3. Data Interpolation
Pandas의 `Linear Interpolation`을 활용하여 Detection이 튀거나 누락된 프레임을 부드럽게 보간하여 궤적의 연속성 확보.

## ⚖️ Trade-offs
* **Linear Mapping vs Homography**: 복잡한 호모그래피 변환 대신 **Linear Mapping**을 채택했습니다. 정밀한 물리적 거리(cm 단위)보다는 **Zone 14 점유 등 전술적 흐름 파악**에 최적화하여 연산 속도를 확보했습니다.
* **Heuristic Stitching**: 딥러닝 기반의 Feature Matching(Re-ID) 대신 거리 기반 매칭을 사용했기에, 매우 혼잡한 상황에서는 오매칭 가능성이 존재합니다. (하지만 일반적인 경기 흐름에서는 충분한 성능 입증)

## 🛠️ Tech Stack
* **Core**: Python, OpenCV, Pandas
* **AI/ML**: YOLOv10 (Object Detection + BoT-SORT Tracking)
* **Visualization**: Matplotlib, Seaborn (Heatmap & KDE Plot)

## 🚀 Usage
```
# Install dependencies
pip install -r requirements.txt
```

# Run Tracking (x, y is initial coordinate of target player)
python football_tracker.py --video match_clip.mp4 --tx 300 --ty 360 --output result.mp4

### 3. 📦 requirements.txt (필수 라이브러리)
```
opencv-python
numpy
pandas
ultralytics
matplotlib
seaborn
scipy
lapx
```
