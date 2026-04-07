# 노트북 웹캠 카메라 보정 및 렌즈 왜곡 교정

OpenCV를 활용한 노트북 웹캠의 카메라 보정 및 렌즈 왜곡 교정 파이프라인입니다.  
웹캠 영상에 fisheye 왜곡을 실시간 적용하여 녹화하고, 이를 캘리브레이션 입력으로 사용해 내부 파라미터(fx, fy, cx, cy)와 왜곡 계수(k1, k2, p1, p2)를 추출한 뒤 왜곡이 교정된 영상을 출력합니다.

---

## 파일 구성

| 파일 | 설명 |
|------|------|
| `gui.py` | **그래픽 UI** — 통합 앱 (권장) |
| `camera_calibration.py` | 영상에서 체스보드 코너를 검출하고 카메라 내부 파라미터 계산 |
| `distortion_correction.py` | 보정 결과를 영상 또는 이미지에 적용하여 왜곡 교정 |
| `generate_chessboard.py` | A4 출력용 체스보드 캘리브레이션 타겟 생성 (CLI 전용) |

---

## 요구 사항

```
Python >= 3.10
opencv-python >= 4.8
numpy >= 1.24
customtkinter >= 5.0
Pillow >= 9.0
```

설치:

```bash
pip install opencv-python numpy customtkinter Pillow
```

---

## 사용법 — GUI (권장)

```bash
python gui.py
```

앱을 실행하면 세 개의 탭이 열립니다.

### 탭 1 — 웹캠 왜곡 영상 녹화
<img width="2555" height="1475" alt="image" src="https://github.com/user-attachments/assets/a63be508-fdeb-47f3-8dbb-8b653fcecbdc" />

1. **Camera index** 설정 (노트북 내장 웹캠: 기본값 `0`)
2. **k1 슬라이더**로 fisheye 왜곡 강도 조절 (0.0 ~ 2.0, 기본값 0.5)
3. **[▶ 미리보기 시작]** 클릭 → 왜곡된 웹캠 영상 실시간 확인
4. 체스보드(종이 인쇄 또는 태블릿 화면)를 다양한 각도로 보여주면서 **[● 녹화 시작]**
5. 충분한 시점을 확보한 후 **[■ 중지]** → 영상 저장 및 탭2 경로 자동 입력

> **팁:** 체스보드가 구부러지지 않도록 주의하세요. 아이패드 등 태블릿 화면에 띄우는 것이 가장 편리합니다.

### 탭 2 — 카메라 보정
<img width="2559" height="1401" alt="image" src="https://github.com/user-attachments/assets/b00fe9ac-054c-41bd-b84e-9aa11fe01c52" />

1. 탭1 녹화 완료 시 영상 경로가 자동으로 입력됨 (또는 **파일 선택…** 으로 직접 선택)
2. 내부 코너 수(열 × 행), 사각형 크기(mm), 프레임 간격 설정
3. **[▶ Run Calibration]** 클릭
4. 완료되면 결과 테이블에 fx, fy, cx, cy, k1~k2, RMSE 표시
5. 보정 데이터는 `results/calibration.npz`에 저장됨

> **팁:** 다양한 각도의 시점을 최소 20개 이상 확보해야 안정적인 결과를 얻을 수 있습니다.

### 탭 3 — 왜곡 교정
<img width="2543" height="1485" alt="image" src="https://github.com/user-attachments/assets/b74b4205-9a09-42be-82ca-22aec5d917c6" />


1. **영상** 또는 **이미지** 모드 선택
2. **파일 선택…** 클릭하여 입력 파일 선택
3. **Alpha 슬라이더** 조정 (0 = 검은 테두리 제거, 1 = 전체 픽셀 유지)
4. **[▶ Run Correction]** 클릭
5. 앱 내에서 보정 전/후 비교 미리보기 표시
6. 교정된 영상과 비교 이미지가 `results/` 폴더에 저장됨

---

## 사용법 — 명령줄

### 1단계 — 캘리브레이션 타겟 준비

체스보드를 직접 출력하려면:

```bash
python generate_chessboard.py --cols 10 --rows 7 --square_mm 25 --out chessboard.png
```

`chessboard.png`를 **100% 크기**로 인쇄하고 평평한 딱딱한 표면에 부착합니다.  
또는 태블릿(아이패드 등) 화면에 이미지를 띄워 사용하면 더 편리합니다.

### 2단계 — 카메라 보정 실행

```bash
python camera_calibration.py \
    --video data/chessboard.mp4 \
    --cols 9 --rows 6 \
    --square_mm 25 \
    --step 5 \
    --min_frames 20 \
    --save_frames \
    --output results/calibration.npz
```

| 옵션 | 설명 |
|------|------|
| `--video` | 입력 영상 경로 |
| `--cols` | 가로 방향 내부 코너 수 (사각형 수 - 1) |
| `--rows` | 세로 방향 내부 코너 수 (사각형 수 - 1) |
| `--square_mm` | 실측 사각형 크기 (mm) |
| `--step` | N번째 프레임마다 샘플링 |
| `--min_frames` | 최소 허용 검출 프레임 수 |
| `--save_frames` | 코너 검출 프레임을 `results/calibration_frames/`에 저장 |
| `--no_display` | 헤드리스 실행 (GUI 창 없음) |

### 3단계 — 렌즈 왜곡 교정 적용

```bash
python distortion_correction.py \
    --video data/chessboard.mp4 \
    --calib results/calibration.npz \
    --alpha 0 \
    --out_video results/undistorted.mp4 \
    --out_compare results/comparison.jpg
```

| 옵션 | 설명 |
|------|------|
| `--alpha 0` | 검은 테두리 제거 (유효 픽셀만 표시) |
| `--alpha 1` | 전체 픽셀 유지 (가장자리에 검은 테두리 발생 가능) |
| `--image` | 영상 대신 단일 이미지 사용 |

---

## 보정 결과

> 카메라: **노트북 내장 웹캠**  
> 해상도: `640 x 480` px

### 내부 파라미터

| 파라미터 | 값 |
|----------|----|
| fx | 576.5205 px |
| fy | 579.6238 px |
| cx | 326.5690 px |
| cy | 215.1782 px |

### 왜곡 계수

| k1 | k2 | p1 | p2 |
|----|----|----|----|
| 1.722556 | -2.634378 | -0.075313 | 0.024335 |

### 재투영 오차

| RMSE |
|------|
| 0.5828 px |

---

## 렌즈 왜곡 교정 데모

<img width="1345" height="504" alt="image" src="https://github.com/user-attachments/assets/060e2933-7b73-4bb3-ad7f-0b0851515e62" />

유튜브 시연 영상 링크 : [https://www.youtube.com/watch?v=FB89C6G3c58&feature=youtu.be](https://www.youtube.com/watch?v=YVEs10eS0VM)

*좌: 원본 (fisheye 왜곡 적용됨) | 우: 왜곡 교정됨*

![frame_00875](https://github.com/user-attachments/assets/2f0cab2a-bfe0-478b-bcc6-ae61aa887d6a)
![frame_00860](https://github.com/user-attachments/assets/3464ce5b-8203-4edb-abef-a51648a24538)
![frame_00810](https://github.com/user-attachments/assets/9c995277-1c3c-46e2-9eb0-c20dd4bba890)
![frame_00470](https://github.com/user-attachments/assets/7cbc56d8-0f41-48b4-8517-1c5b129a2e46)
![frame_00445](https://github.com/user-attachments/assets/5f1933eb-99f5-4629-bc8e-6a038378e340)
![frame_00425](https://github.com/user-attachments/assets/7b26f5d7-2b6a-48fd-b281-ac85d70fd58e)
![frame_00330](https://github.com/user-attachments/assets/ef77b345-a2b7-4618-a52e-aafe2c523093)
![frame_00325](https://github.com/user-attachments/assets/67d82d8c-a3d7-4acc-89f3-6a784ee391aa)


---

## 참고 사항

- RMSE < 1.0 px는 일반적으로 허용 가능한 수준이며, < 0.5 px면 우수한 품질입니다.
- 실제 사용할 해상도와 동일한 해상도로 보정을 수행하십시오.
- `results/calibration.npz` 파일에 전체 카메라 행렬과 왜곡 계수가 저장되어 재사용할 수 있습니다.
