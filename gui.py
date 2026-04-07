"""
gui.py
------
Graphical User Interface for the Camera Calibration & Lens Distortion Correction pipeline.

Usage:
    python gui.py

Tabs:
  1. Webcam Recorder       — 웹캠으로 fisheye 왜곡 영상 실시간 녹화 후 저장
  2. Camera Calibration    — detect corners, compute intrinsics, view results
  3. Distortion Correction — apply calibration to a video or image

Requirements:
    pip install opencv-python numpy customtkinter Pillow
"""

import os
import sys
import threading
import queue
import time
import io

import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ── Appearance ────────────────────────────────────────────────────────────────
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT   = "#1f6aa5"
BG_DARK  = "#1a1a2e"
BG_MID   = "#16213e"
BG_CARD  = "#0f3460"
TEXT_DIM = "#8899aa"

RESULT_KEYS = ["fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3", "rmse"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def find_corners(gray, pattern_size):
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH
             | cv2.CALIB_CB_FAST_CHECK
             | cv2.CALIB_CB_NORMALIZE_IMAGE)
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
    if not found:
        return False, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def build_object_points(pattern_size, square_mm):
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm
    return objp


def cv2_to_photoimage(bgr_frame, max_w=None, max_h=None):
    """Convert a BGR OpenCV frame to a CTkImage-compatible PIL Image."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    if max_w or max_h:
        img.thumbnail((max_w or 99999, max_h or 99999), Image.LANCZOS)
    return img


# ── Reusable widgets ──────────────────────────────────────────────────────────

class LogBox(ctk.CTkTextbox):
    """Thread-safe scrollable log console."""

    def __init__(self, master, **kw):
        kw.setdefault("font", ("Consolas", 12))
        kw.setdefault("state", "disabled")
        kw.setdefault("wrap", "word")
        super().__init__(master, **kw)
        self._q = queue.Queue()
        self._poll()

    def log(self, msg: str, tag: str = ""):
        self._q.put((msg, tag))

    def _poll(self):
        try:
            while True:
                msg, tag = self._q.get_nowait()
                self.configure(state="normal")
                self.insert("end", msg + "\n")
                self.see("end")
                self.configure(state="disabled")
        except queue.Empty:
            pass
        self.after(60, self._poll)

    def clear(self):
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.configure(state="disabled")


class FileRow(ctk.CTkFrame):
    """Label + Entry + Browse button row for file selection."""

    def __init__(self, master, label: str, filetypes=None, mode="open", **kw):
        super().__init__(master, fg_color="transparent", **kw)
        self._mode = mode
        self._types = filetypes or [("All files", "*.*")]
        ctk.CTkLabel(self, text=label, width=140, anchor="w").pack(side="left")
        self.var = ctk.StringVar()
        self._entry = ctk.CTkEntry(self, textvariable=self.var, width=380)
        self._entry.pack(side="left", padx=(4, 6))
        ctk.CTkButton(self, text="Browse…", width=90,
                      command=self._browse).pack(side="left")

    def _browse(self):
        if self._mode == "save":
            path = filedialog.asksaveasfilename(filetypes=self._types,
                                               defaultextension=self._types[0][1])
        else:
            path = filedialog.askopenfilename(filetypes=self._types)
        if path:
            self.var.set(path)

    @property
    def path(self):
        return self.var.get().strip()


class ParamRow(ctk.CTkFrame):
    """Label + Entry row for a numeric parameter."""

    def __init__(self, master, label: str, default, width=100, **kw):
        super().__init__(master, fg_color="transparent", **kw)
        ctk.CTkLabel(self, text=label, width=180, anchor="w").pack(side="left")
        self.var = ctk.StringVar(value=str(default))
        ctk.CTkEntry(self, textvariable=self.var, width=width).pack(side="left")

    def get_int(self):
        return int(self.var.get())

    def get_float(self):
        return float(self.var.get())


class ResultTable(ctk.CTkFrame):
    """Grid of label pairs showing calibration results."""

    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._labels = {}
        fields = [
            ("fx (px)",  "fy (px)",  "cx (px)",  "cy (px)"),
            ("k1",       "k2",       "p1",        "p2"),
            ("k3",       "RMSE (px)", "",          ""),
        ]
        for r, row in enumerate(fields):
            for c, name in enumerate(row):
                if not name:
                    continue
                key = name.split()[0].lower()
                ctk.CTkLabel(self, text=name, text_color=TEXT_DIM,
                             font=("Segoe UI", 12)).grid(
                    row=r * 2, column=c, padx=18, pady=(8, 0), sticky="w")
                lbl = ctk.CTkLabel(self, text="—",
                                   font=("Consolas", 14, "bold"))
                lbl.grid(row=r * 2 + 1, column=c, padx=18, pady=(0, 8), sticky="w")
                self._labels[key] = lbl

    def update(self, values: dict):
        for k, v in values.items():
            key = k.lower()
            if key in self._labels:
                self._labels[key].configure(text=f"{v:.5f}" if isinstance(v, float) else str(v))

    def reset(self):
        for lbl in self._labels.values():
            lbl.configure(text="—")


# ── Tab 1: Chessboard Generator ───────────────────────────────────────────────

class WebcamRecorderTab(ctk.CTkFrame):
    """Tab 1: 웹캠 실시간 fisheye 왜곡 적용 및 영상 녹화."""

    def __init__(self, master, calib_tab=None):
        super().__init__(master, fg_color="transparent")
        # 스레드 제어
        self._stop_flag       = threading.Event()
        self._thread          = None
        self._running         = False
        # 녹화 상태
        self._recording       = False
        self._writer          = None
        self._writer_lock     = threading.Lock()
        # 왜곡 맵 캐시
        self._dist_cache      = {}
        # 미리보기 큐 포화 방지
        self._preview_pending = False
        # 탭2 참조 (녹화 완료 시 경로 자동 입력)
        self._calib_tab       = calib_tab
        self._build()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build(self):
        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, corner_radius=12)
        ctrl.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(ctrl, text="웹캠 왜곡 영상 녹화",
                     font=("Segoe UI", 18, "bold")).pack(
            anchor="w", padx=20, pady=(16, 8))

        # 카메라 인덱스 + k1 슬라이더
        param_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        param_row.pack(fill="x", padx=20, pady=4)

        ctk.CTkLabel(param_row, text="Camera index",
                     text_color=TEXT_DIM, width=110, anchor="w").pack(side="left")
        self._cam_idx_var = ctk.StringVar(value="0")
        ctk.CTkEntry(param_row, textvariable=self._cam_idx_var,
                     width=55).pack(side="left", padx=(0, 28))

        ctk.CTkLabel(param_row, text="k1 (barrel/fisheye)",
                     text_color=TEXT_DIM, width=150, anchor="w").pack(side="left")
        self._k1_var = ctk.DoubleVar(value=0.5)
        self._k1_lbl = ctk.CTkLabel(param_row, text="0.50", width=48)
        ctk.CTkSlider(
            param_row, from_=0.0, to=2.0, number_of_steps=40,
            variable=self._k1_var,
            command=lambda v: (
                self._k1_lbl.configure(text=f"{v:.2f}"),
                self._invalidate_dist_cache()
            )
        ).pack(side="left", padx=8)
        self._k1_lbl.pack(side="left")

        # 출력 경로
        out_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        out_row.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(out_row, text="Output .mp4",
                     text_color=TEXT_DIM, width=110, anchor="w").pack(side="left")
        self._out_var = ctk.StringVar(value="recordings/distorted.mp4")
        ctk.CTkEntry(out_row, textvariable=self._out_var,
                     width=380).pack(side="left", padx=(0, 6))
        ctk.CTkButton(out_row, text="Browse…", width=90,
                      command=self._browse_output).pack(side="left")

        # 버튼 행
        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(8, 14))

        self._start_btn = ctk.CTkButton(
            btn_row, text="▶  미리보기 시작",
            command=self._start_preview, width=160)
        self._start_btn.pack(side="left")

        self._rec_btn = ctk.CTkButton(
            btn_row, text="●  녹화 시작",
            command=self._start_recording, width=130,
            fg_color="#7a1f1f", state="disabled")
        self._rec_btn.pack(side="left", padx=10)

        self._stop_btn = ctk.CTkButton(
            btn_row, text="■  중지",
            command=self._stop_preview, width=100,
            fg_color="#aa3333", state="disabled")
        self._stop_btn.pack(side="left")

        self._fps_lbl = ctk.CTkLabel(
            btn_row, text="FPS: --", text_color=TEXT_DIM, width=80)
        self._fps_lbl.pack(side="left", padx=16)

        self._rec_indicator = ctk.CTkLabel(
            btn_row, text="", text_color="#ff5555", width=100)
        self._rec_indicator.pack(side="left")

        # ── 실시간 미리보기 ────────────────────────────────────────────────────
        prev_frame = ctk.CTkFrame(self, corner_radius=12)
        prev_frame.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        ctk.CTkLabel(prev_frame, text="Live Preview (왜곡 적용)",
                     font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))
        self._preview_lbl = ctk.CTkLabel(
            prev_frame,
            text="미리보기 시작 버튼을 눌러 웹캠을 연결하세요.",
            text_color=TEXT_DIM)
        self._preview_lbl.pack(expand=True)

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self, corner_radius=12)
        log_frame.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(log_frame, text="Log",
                     font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))
        self._log = LogBox(log_frame, height=110)
        self._log.pack(fill="x", padx=12, pady=(0, 12))

    def _browse_output(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".mp4",
            filetypes=[("MP4 video", "*.mp4"), ("All files", "*.*")],
            initialfile="distorted.mp4")
        if path:
            self._out_var.set(path)

    # ── 왜곡 맵 ──────────────────────────────────────────────────────────────

    def _build_distortion_map(self, h: int, w: int, k1: float):
        """해상도·k1이 동일하면 캐시 반환, 아니면 재생성."""
        cache = self._dist_cache
        if (cache.get("h") == h and cache.get("w") == w and
                abs(cache.get("k1", -1) - k1) < 1e-6):
            return cache["map_x"], cache["map_y"]

        cx, cy = w / 2.0, h / 2.0
        f = max(w, h) * 0.7
        ys, xs = np.mgrid[0:h, 0:w]
        xd = (xs - cx) / f
        yd = (ys - cy) / f
        r2 = xd ** 2 + yd ** 2
        factor = 1.0 / (1.0 + k1 * r2)   # k1 > 0 → barrel / fisheye
        map_x = (xd * factor * f + cx).astype(np.float32)
        map_y = (yd * factor * f + cy).astype(np.float32)
        self._dist_cache = {"h": h, "w": w, "k1": k1,
                            "map_x": map_x, "map_y": map_y}
        return map_x, map_y

    def _invalidate_dist_cache(self):
        self._dist_cache = {}

    # ── 제어 메서드 ───────────────────────────────────────────────────────────

    def _start_preview(self):
        if self._running:
            return
        try:
            cam_idx = int(self._cam_idx_var.get())
        except ValueError:
            messagebox.showerror("입력 오류", "Camera index는 정수여야 합니다.")
            return

        out_path = self._out_var.get().strip() or "recordings/distorted.mp4"

        self._running = True
        self._stop_flag = threading.Event()
        self._dist_cache = {}

        self._start_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._rec_btn.configure(state="normal")
        self._log.log(f"[INFO] 미리보기 시작 — 카메라 인덱스: {cam_idx}")

        self._thread = threading.Thread(
            target=self._webcam_loop,
            args=(cam_idx, out_path),
            daemon=True
        )
        self._thread.start()

    def _start_recording(self):
        if not self._running or self._recording:
            return

        cache = self._dist_cache
        if not cache:
            self._log.log("[WARN] 첫 프레임을 기다리는 중… 잠시 후 다시 시도하세요.")
            return

        w, h = cache.get("w", 640), cache.get("h", 480)
        out_path = self._out_var.get().strip() or "recordings/distorted.mp4"
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, 30.0, (w, h))
        if not writer.isOpened():
            self._log.log(f"[ERROR] VideoWriter 열기 실패: {out_path}")
            return

        with self._writer_lock:
            self._writer = writer

        self._recording = True
        self._rec_btn.configure(state="disabled")
        self._rec_indicator.configure(text="● 녹화 중")
        self._log.log(f"[INFO] 녹화 시작 → {out_path}  ({w}×{h})")

    def _stop_preview(self):
        if not self._running:
            return
        self._stop_btn.configure(state="disabled")
        self._rec_btn.configure(state="disabled")
        self._log.log("[INFO] 중지 요청 — 현재 프레임 완료 후 종료합니다…")
        self._stop_flag.set()

    def _on_loop_ended(self):
        """웹캠 루프 종료 후 GUI 스레드에서 호출."""
        self._running         = False
        self._recording       = False
        self._writer          = None
        self._preview_pending = False
        self._start_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._rec_btn.configure(state="disabled")
        self._rec_indicator.configure(text="")
        self._fps_lbl.configure(text="FPS: --")
        self._preview_lbl.configure(
            image="",
            text="미리보기 시작 버튼을 눌러 웹캠을 연결하세요.")

    # ── 웹캠 루프 (백그라운드 스레드) ────────────────────────────────────────

    def _webcam_loop(self, cam_idx: int, out_path: str):
        log = self._log.log
        cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            log(f"[ERROR] 카메라 인덱스 {cam_idx}를 열 수 없습니다.")
            self.after(0, self._on_loop_ended)
            return

        fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log(f"[INFO] 웹캠 열림 — 인덱스:{cam_idx}  해상도:{w}×{h}  FPS:{fps_src:.1f}")

        # 디스플레이 크기 계산 (860×440 안에 비율 유지)
        DISP_W, DISP_H = 860, 440
        scale  = min(DISP_W / w, DISP_H / h)
        disp_w = int(w * scale)
        disp_h = int(h * scale)

        frame_count = 0
        fps_display = 0.0
        t_fps = time.perf_counter()

        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                log("[WARN] 프레임 읽기 실패, 재시도 중…")
                time.sleep(0.05)
                continue

            # 왜곡 적용
            k1 = self._k1_var.get()
            map_x, map_y = self._build_distortion_map(h, w, k1)
            distorted = cv2.remap(frame, map_x, map_y,
                                  cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)

            # 녹화
            with self._writer_lock:
                if self._writer is not None:
                    self._writer.write(distorted)

            # FPS 측정 (0.5초 주기)
            frame_count += 1
            elapsed = time.perf_counter() - t_fps
            if elapsed >= 0.5:
                fps_display = frame_count / elapsed
                frame_count = 0
                t_fps = time.perf_counter()

            # 미리보기 업데이트 (큐 포화 방지)
            if not self._preview_pending:
                self._preview_pending = True
                dist_copy = distorted.copy()
                fps_val   = fps_display
                self.after(0, lambda f=dist_copy, fps=fps_val:
                           self._update_preview(f, fps, disp_w, disp_h))

        # 루프 종료 정리
        cap.release()
        saved_path = None
        with self._writer_lock:
            if self._writer is not None:
                self._writer.release()
                self._writer = None
                saved_path = out_path

        if saved_path:
            log(f"[OK] 녹화 저장 완료 → {saved_path}")
            if self._calib_tab is not None:
                self.after(0, lambda p=saved_path:
                           self._calib_tab._video_row.var.set(p))

        log("[INFO] 웹캠 루프 종료.")
        self.after(0, self._on_loop_ended)

    # ── GUI 스레드 전용 업데이트 ──────────────────────────────────────────────

    def _update_preview(self, frame_bgr, fps: float, disp_w: int, disp_h: int):
        try:
            img = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img,
                                   size=(disp_w, disp_h))
            self._preview_lbl.configure(image=ctk_img, text="")
            self._preview_lbl._image = ctk_img  # GC 방지
            if fps > 0:
                self._fps_lbl.configure(text=f"FPS: {fps:.1f}")
        finally:
            self._preview_pending = False


# ── Tab 2: Camera Calibration ─────────────────────────────────────────────────

class CalibrationTab(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._running   = False
        self._thread    = None
        self._result    = {}
        self._build()

    def _build(self):
        # ── Top: file + params ────────────────────────────────────────────────
        top = ctk.CTkFrame(self, corner_radius=12)
        top.pack(fill="x", padx=20, pady=(20, 8))

        ctk.CTkLabel(top, text="Camera Calibration",
                     font=("Segoe UI", 18, "bold")).pack(
            anchor="w", padx=20, pady=(14, 8))

        self._video_row = FileRow(top, "Video file",
                                  [("Video", "*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV"),
                                   ("All", "*.*")])
        self._video_row.pack(fill="x", padx=20, pady=4)

        self._out_row = FileRow(top, "Output .npz",
                                [("NumPy archive", "*.npz"), ("All", "*.*")],
                                mode="save")
        self._out_row.var.set("results/calibration.npz")
        self._out_row.pack(fill="x", padx=20, pady=4)

        # Parameters grid
        pg = ctk.CTkFrame(top, fg_color="transparent")
        pg.pack(fill="x", padx=20, pady=(8, 14))

        params = [
            ("Inner corners (cols)", "9"),
            ("Inner corners (rows)", "6"),
            ("Square size (mm)",     "25"),
            ("Frame step",           "5"),
            ("Min frames",           "20"),
        ]
        self._pvars = []
        for i, (lbl, val) in enumerate(params):
            col = i % 3
            row = (i // 3) * 2
            ctk.CTkLabel(pg, text=lbl, text_color=TEXT_DIM).grid(
                row=row, column=col * 2, padx=(0 if col == 0 else 20, 6), pady=4, sticky="e")
            var = ctk.StringVar(value=val)
            ctk.CTkEntry(pg, textvariable=var, width=85).grid(
                row=row, column=col * 2 + 1, padx=(0, 6), pady=4, sticky="w")
            self._pvars.append(var)

        self._save_frames = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(pg, text="Save detected-corner frames",
                        variable=self._save_frames).grid(
            row=4, column=0, columnspan=3, padx=0, pady=(4, 0), sticky="w")

        # Buttons
        btn_row = ctk.CTkFrame(top, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(0, 14))

        self._run_btn = ctk.CTkButton(btn_row, text="▶  Run Calibration",
                                      command=self._start, width=180)
        self._run_btn.pack(side="left")

        self._stop_btn = ctk.CTkButton(btn_row, text="■  Stop",
                                       command=self._stop, width=100,
                                       fg_color="#aa3333", state="disabled")
        self._stop_btn.pack(side="left", padx=12)

        ctk.CTkButton(btn_row, text="🔍  Test Frame",
                      command=self._test_frame, width=120,
                      fg_color="#2d6a4f").pack(side="left", padx=4)

        self._prog = ctk.CTkProgressBar(btn_row, width=220, mode="indeterminate")
        self._prog.pack(side="left", padx=8)

        # ── Preview (test frame) ───────────────────────────────────────────────
        self._test_lbl = ctk.CTkLabel(top, text="", text_color=TEXT_DIM,
                                      font=("Segoe UI", 12))
        self._test_lbl.pack(anchor="w", padx=20, pady=(0, 2))

        self._test_preview = ctk.CTkLabel(top, text="")
        self._test_preview.pack(pady=(0, 6))

        # ── Results table ─────────────────────────────────────────────────────
        res_frame = ctk.CTkFrame(self, corner_radius=12)
        res_frame.pack(fill="x", padx=20, pady=(0, 8))
        ctk.CTkLabel(res_frame, text="Calibration Results",
                     font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))
        self._result_table = ResultTable(res_frame)
        self._result_table.pack(fill="x", padx=10, pady=(0, 10))

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self, corner_radius=12)
        log_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        ctk.CTkLabel(log_frame, text="Log",
                     font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))
        self._log = LogBox(log_frame, height=160)
        self._log.pack(fill="both", expand=True, padx=12, pady=(0, 12))

    def _test_frame(self):
        """중간 프레임 하나를 뽑아서 코너 검출을 시도하고 결과를 미리보기로 표시."""
        video = self._video_row.path
        if not video or not os.path.isfile(video):
            messagebox.showwarning("Input Required", "먼저 동영상 파일을 선택하세요.")
            return
        try:
            cols = int(self._pvars[0].get())
            rows = int(self._pvars[1].get())
        except ValueError:
            return
        pattern_size = (cols, rows)

        cap   = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        # 중간 프레임으로 이동
        cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
        ret, frame = cap.read()
        cap.release()

        if not ret or frame is None:
            messagebox.showerror("Error", "프레임을 읽을 수 없습니다.")
            return

        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size)

        vis = frame.copy()
        if found:
            cv2.drawChessboardCorners(vis, pattern_size, corners, True)
            status = f"✅  코너 검출 성공! ({cols}×{rows} inner corners)  →  캘리브레이션을 실행하세요."
            color  = "#aaffaa"
        else:
            # 검출 실패 시 — 이미지에 안내 텍스트 표시
            cv2.putText(vis, f"Not detected ({cols}x{rows})", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 255), 3, cv2.LINE_AA)
            status = (f"❌  코너 미검출 (inner corners {cols}×{rows}) — "
                      "체스보드가 잘 보이는지, cols/rows 값이 맞는지 확인하세요.")
            color  = "#ffaaaa"

        self._test_lbl.configure(text=status, text_color=color)
        pil     = cv2_to_photoimage(vis, max_w=860, max_h=280)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                                size=(pil.width, pil.height))
        self._test_preview.configure(image=ctk_img, text="")
        self._test_preview._image = ctk_img

    def _start(self):
        if self._running:
            return
        video = self._video_row.path
        out   = self._out_row.path or "results/calibration.npz"
        if not video:
            messagebox.showwarning("Input Required", "Select a video file first.")
            return
        if not os.path.isfile(video):
            messagebox.showerror("File Not Found", f"Video not found:\n{video}")
            return
        try:
            cols       = int(self._pvars[0].get())
            rows       = int(self._pvars[1].get())
            square_mm  = float(self._pvars[2].get())
            step       = int(self._pvars[3].get())
            min_frames = int(self._pvars[4].get())
        except ValueError as e:
            messagebox.showerror("Input Error", f"Invalid parameter: {e}")
            return

        self._log.clear()
        self._result_table.reset()
        self._running = True
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._prog.start()
        self._stop_flag = threading.Event()

        self._thread = threading.Thread(
            target=self._run_calibration,
            args=(video, out, cols, rows, square_mm, step, min_frames,
                  self._save_frames.get()),
            daemon=True
        )
        self._thread.start()

    def _stop(self):
        if self._stop_flag:
            self._stop_flag.set()
            self._log.log("[INFO] Stop requested — finishing current frame…")

    def _finish(self):
        self._running = False
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._prog.stop()

    def _run_calibration(self, video, out, cols, rows, square_mm, step, min_frames, save_frames):
        log = self._log.log
        pattern_size = (cols, rows)
        objp = build_object_points(pattern_size, square_mm)

        obj_points = []
        img_points = []

        cap = cv2.VideoCapture(video)
        if not cap.isOpened():
            log(f"[ERROR] Cannot open video: {video}")
            self.after(0, self._finish)
            return

        total   = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps     = cap.get(cv2.CAP_PROP_FPS) or 30.0
        vw      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        img_size = (vw, vh)

        log(f"[INFO] Video  : {video}")
        log(f"       Size   : {vw} x {vh}  |  FPS: {fps:.1f}  |  Frames: {total}")
        log(f"[INFO] Pattern: {cols}x{rows} inner corners | Square: {square_mm} mm")
        log(f"[INFO] Sampling every {step} frames …")

        out_dir = os.path.dirname(out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

        frames_dir = None
        if save_frames:
            frames_dir = os.path.join(out_dir or "results", "calibration_frames")
            os.makedirs(frames_dir, exist_ok=True)

        frame_idx   = 0
        detect_cnt  = 0
        saved_cnt   = 0

        log(f"[INFO] 프레임 스캔 시작 — 총 {total}프레임, {step}프레임마다 검사")
        log(f"       체스보드가 잘 보이지 않는 프레임은 건너뜁니다.")
        log("")

        while not self._stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % step != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = find_corners(gray, pattern_size)

            if found:
                obj_points.append(objp)
                img_points.append(corners)
                detect_cnt += 1

                vis = frame.copy()
                cv2.drawChessboardCorners(vis, pattern_size, corners, found)
                if frames_dir and saved_cnt < 50:
                    cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg"), vis)
                    saved_cnt += 1

                if detect_cnt <= 5 or detect_cnt % 5 == 0:
                    log(f"  [Frame {frame_idx:5d}/{total}]  ✓ Corner detected  (total: {detect_cnt})")
            else:
                # 50프레임 스캔마다 진행상황 표시 (미검출 시에도)
                scanned = frame_idx // step
                if scanned % 50 == 0 and scanned > 0:
                    pct = frame_idx / max(total, 1) * 100
                    log(f"  [Frame {frame_idx:5d}/{total}]  Scanning…  ({pct:.1f}%)  detected so far: {detect_cnt}")

            frame_idx += 1

        cap.release()
        log(f"[INFO] Scan done — {detect_cnt} frames with corners detected")

        if detect_cnt < min_frames:
            log(f"[ERROR] Only {detect_cnt} frames (need ≥ {min_frames}). Record more viewpoints.")
            self.after(0, self._finish)
            return

        log(f"\n[INFO] Running cv2.calibrateCamera() on {detect_cnt} frames …")
        rms, cam_mat, dist, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, img_size, None, None
        )

        d  = dist.ravel()
        fx, fy = cam_mat[0, 0], cam_mat[1, 1]
        cx, cy = cam_mat[0, 2], cam_mat[1, 2]
        k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
        k3 = d[4] if len(d) > 4 else 0.0

        log("\n" + "=" * 52)
        log("  Calibration Results")
        log("=" * 52)
        log(f"  fx = {fx:.4f} px      fy = {fy:.4f} px")
        log(f"  cx = {cx:.4f} px      cy = {cy:.4f} px")
        log(f"  k1 = {k1:.6f}   k2 = {k2:.6f}")
        log(f"  p1 = {p1:.6f}   p2 = {p2:.6f}")
        log(f"  k3 = {k3:.6f}")
        log(f"  RMSE = {rms:.4f} px")
        if rms < 0.5:
            log("  [Excellent] RMSE < 0.5 px")
        elif rms < 1.0:
            log("  [Good] RMSE < 1.0 px")
        else:
            log("  [Warning] RMSE ≥ 1.0 px — recapture more diverse viewpoints")
        log("=" * 52)

        np.savez(out, camera_matrix=cam_mat, dist_coeffs=dist,
                 img_size=np.array(img_size), rms=np.array(rms),
                 square_mm=np.array(square_mm))
        log(f"\n[OK] Saved → {out}")
        if frames_dir:
            log(f"[OK] Corner frames → {frames_dir}/  ({saved_cnt} files)")

        self._result = dict(fx=fx, fy=fy, cx=cx, cy=cy,
                            k1=k1, k2=k2, p1=p1, p2=p2, k3=k3, rmse=rms)
        self.after(0, lambda: self._result_table.update(self._result))
        self.after(0, self._finish)


# ── Tab 3: Distortion Correction ──────────────────────────────────────────────

class CorrectionTab(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._running    = False
        self._stop_flag  = threading.Event()
        self._build()

    def _build(self):
        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, corner_radius=12)
        ctrl.pack(fill="x", padx=20, pady=(20, 8))

        ctk.CTkLabel(ctrl, text="Lens Distortion Correction",
                     font=("Segoe UI", 18, "bold")).pack(
            anchor="w", padx=20, pady=(14, 8))

        # Input mode toggle
        mode_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        mode_row.pack(fill="x", padx=20, pady=4)
        ctk.CTkLabel(mode_row, text="Input mode", text_color=TEXT_DIM, width=140,
                     anchor="w").pack(side="left")
        self._mode = ctk.StringVar(value="video")
        ctk.CTkRadioButton(mode_row, text="Video", variable=self._mode, value="video",
                           command=self._toggle_mode).pack(side="left", padx=(0, 20))
        ctk.CTkRadioButton(mode_row, text="Image", variable=self._mode, value="image",
                           command=self._toggle_mode).pack(side="left")

        self._video_row = FileRow(ctrl, "Video file",
                                  [("Video", "*.mp4 *.avi *.mov *.MP4 *.AVI *.MOV"),
                                   ("All", "*.*")])
        self._video_row.pack(fill="x", padx=20, pady=4)

        self._image_row = FileRow(ctrl, "Image file",
                                  [("Image", "*.jpg *.jpeg *.png *.bmp"),
                                   ("All", "*.*")])
        self._image_row.pack(fill="x", padx=20, pady=4)
        self._image_row.pack_forget()

        self._calib_row = FileRow(ctrl, "Calibration .npz",
                                  [("NumPy archive", "*.npz"), ("All", "*.*")])
        self._calib_row.var.set("results/calibration.npz")
        self._calib_row.pack(fill="x", padx=20, pady=4)

        self._out_vid_row = FileRow(ctrl, "Output video",
                                    [("MP4", "*.mp4"), ("All", "*.*")], mode="save")
        self._out_vid_row.var.set("results/undistorted.mp4")
        self._out_vid_row.pack(fill="x", padx=20, pady=4)

        self._out_cmp_row = FileRow(ctrl, "Comparison image",
                                    [("JPEG", "*.jpg"), ("PNG", "*.png")], mode="save")
        self._out_cmp_row.var.set("results/comparison.jpg")
        self._out_cmp_row.pack(fill="x", padx=20, pady=4)

        # Alpha slider
        alpha_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        alpha_row.pack(fill="x", padx=20, pady=(8, 4))
        ctk.CTkLabel(alpha_row, text="Alpha (0=crop, 1=keep all)",
                     text_color=TEXT_DIM, width=210, anchor="w").pack(side="left")
        self._alpha_var = ctk.DoubleVar(value=0.0)
        self._alpha_lbl = ctk.CTkLabel(alpha_row, text="0.0", width=40)
        sld = ctk.CTkSlider(alpha_row, from_=0, to=1, number_of_steps=20,
                            variable=self._alpha_var,
                            command=lambda v: self._alpha_lbl.configure(text=f"{v:.2f}"))
        sld.pack(side="left", padx=8)
        self._alpha_lbl.pack(side="left")

        # Buttons
        btn_row = ctk.CTkFrame(ctrl, fg_color="transparent")
        btn_row.pack(fill="x", padx=20, pady=(8, 14))
        self._run_btn = ctk.CTkButton(btn_row, text="▶  Run Correction",
                                      command=self._start, width=180)
        self._run_btn.pack(side="left")
        self._stop_btn = ctk.CTkButton(btn_row, text="■  Stop",
                                       command=self._stop, width=100,
                                       fg_color="#aa3333", state="disabled")
        self._stop_btn.pack(side="left", padx=12)
        self._prog = ctk.CTkProgressBar(btn_row, width=260, mode="indeterminate")
        self._prog.pack(side="left", padx=8)

        # ── Preview ───────────────────────────────────────────────────────────
        prev_frame = ctk.CTkFrame(self, corner_radius=12)
        prev_frame.pack(fill="both", expand=True, padx=20, pady=(0, 8))
        ctk.CTkLabel(prev_frame, text="Before / After Comparison",
                     font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))
        self._preview_lbl = ctk.CTkLabel(prev_frame,
                                         text="Run correction to see the comparison preview.",
                                         text_color=TEXT_DIM)
        self._preview_lbl.pack(expand=True)

        # ── Log ───────────────────────────────────────────────────────────────
        log_frame = ctk.CTkFrame(self, corner_radius=12)
        log_frame.pack(fill="x", padx=20, pady=(0, 20))
        ctk.CTkLabel(log_frame, text="Log", font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))
        self._log = LogBox(log_frame, height=120)
        self._log.pack(fill="x", padx=12, pady=(0, 12))

    def _toggle_mode(self):
        if self._mode.get() == "video":
            self._image_row.pack_forget()
            self._video_row.pack(fill="x", padx=20, pady=4,
                                  after=self._image_row.master.winfo_children()[2])
            self._out_vid_row.pack(fill="x", padx=20, pady=4)
        else:
            self._video_row.pack_forget()
            self._out_vid_row.pack_forget()
            self._image_row.pack(fill="x", padx=20, pady=4)

    def _start(self):
        if self._running:
            return
        calib = self._calib_row.path or "results/calibration.npz"
        alpha = self._alpha_var.get()

        if self._mode.get() == "video":
            inp = self._video_row.path
            out_vid = self._out_vid_row.path or "results/undistorted.mp4"
        else:
            inp = self._image_row.path
            out_vid = None

        out_cmp = self._out_cmp_row.path or "results/comparison.jpg"

        if not inp:
            messagebox.showwarning("Input Required", "Select an input file first.")
            return
        if not os.path.isfile(inp):
            messagebox.showerror("File Not Found", f"Input not found:\n{inp}")
            return
        if not os.path.isfile(calib):
            messagebox.showerror("Calibration Missing",
                                 f"Calibration file not found:\n{calib}\n\n"
                                 "Run Camera Calibration first.")
            return

        self._log.clear()
        self._running = True
        self._stop_flag.clear()
        self._run_btn.configure(state="disabled")
        self._stop_btn.configure(state="normal")
        self._prog.start()

        threading.Thread(
            target=self._run_correction,
            args=(inp, calib, alpha, out_vid, out_cmp),
            daemon=True
        ).start()

    def _stop(self):
        self._stop_flag.set()
        self._log.log("[INFO] Stop requested…")

    def _finish(self):
        self._running = False
        self._run_btn.configure(state="normal")
        self._stop_btn.configure(state="disabled")
        self._prog.stop()

    def _run_correction(self, inp, calib, alpha, out_vid, out_cmp):
        log = self._log.log

        # Load calibration
        data       = np.load(calib)
        cam_mat    = data["camera_matrix"]
        dist       = data["dist_coeffs"]
        calib_size = tuple(data["img_size"])
        rms        = float(data["rms"])

        d  = dist.ravel()
        log("=" * 50)
        log("  Loaded Calibration Parameters")
        log("=" * 50)
        log(f"  fx={cam_mat[0,0]:.2f}  fy={cam_mat[1,1]:.2f}  "
            f"cx={cam_mat[0,2]:.2f}  cy={cam_mat[1,2]:.2f}")
        log(f"  k1={d[0]:.5f}  k2={d[1]:.5f}  p1={d[2]:.5f}  p2={d[3]:.5f}")
        log(f"  RMSE: {rms:.4f} px")
        log("=" * 50)

        is_video = self._mode.get() == "video"

        # Determine image size
        if is_video:
            cap    = cv2.VideoCapture(inp)
            vw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            vh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            img_size = (vw, vh)
        else:
            frame = cv2.imread(inp)
            if frame is None:
                log(f"[ERROR] Cannot read image: {inp}")
                self.after(0, self._finish)
                return
            img_size = (frame.shape[1], frame.shape[0])

        # Build maps
        new_cam, roi = cv2.getOptimalNewCameraMatrix(
            cam_mat, dist, img_size, alpha, img_size)
        map1, map2 = cv2.initUndistortRectifyMap(
            cam_mat, dist, None, new_cam, img_size, cv2.CV_16SC2)

        log(f"\n[INFO] alpha={alpha:.2f}  ({'crop' if alpha == 0 else 'keep all'})")
        log(f"  New fx'={new_cam[0,0]:.2f}  fy'={new_cam[1,1]:.2f}  "
            f"cx'={new_cam[0,2]:.2f}  cy'={new_cam[1,2]:.2f}")

        os.makedirs(os.path.dirname(out_cmp) or ".", exist_ok=True)

        snapshot_done = False

        if is_video:
            os.makedirs(os.path.dirname(out_vid) or ".", exist_ok=True)
            compare_w = img_size[0] * 2 + 4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_vid, fourcc, fps, (compare_w, img_size[1]))
            log(f"\n[INFO] Processing {total} frames …")

            frame_idx = 0
            snap_idx  = total // 2

            while not self._stop_flag.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                corr = cv2.remap(frame, map1, map2,
                                 cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
                cmp  = self._make_comparison(frame, corr)
                if cmp.shape[1] == compare_w and cmp.shape[0] == img_size[1]:
                    writer.write(cmp)

                if frame_idx >= snap_idx and not snapshot_done:
                    cv2.imwrite(out_cmp, cmp)
                    snapshot_done = True
                    log(f"[OK] Comparison snapshot saved → {out_cmp}")
                    self.after(0, lambda c=cmp: self._show_preview(c))

                if frame_idx % 100 == 0:
                    pct = frame_idx / max(total, 1) * 100
                    log(f"  Frame {frame_idx}/{total}  ({pct:.1f}%)")
                frame_idx += 1

            cap.release()
            writer.release()
            log(f"[OK] Corrected video saved → {out_vid}")

        else:
            corr = cv2.remap(frame, map1, map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
            cmp  = self._make_comparison(frame, corr)
            cv2.imwrite(out_cmp, cmp)
            log(f"[OK] Comparison image saved → {out_cmp}")
            self.after(0, lambda c=cmp: self._show_preview(c))

        self.after(0, self._finish)

    @staticmethod
    def _make_comparison(original, corrected):
        sep = np.full((original.shape[0], 4, 3), 128, dtype=np.uint8)
        cmp = np.hstack([original, sep, corrected])
        h   = cmp.shape[0]
        pad = max(10, h // 36)
        fs  = max(0.6, h / 1080)
        tk  = max(1, int(h / 540))
        for text, x in [("Original (distorted)", pad),
                         ("Undistorted", original.shape[1] + 4 + pad)]:
            cv2.putText(cmp, text, (x + 1, pad + 1),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), tk + 1, cv2.LINE_AA)
            cv2.putText(cmp, text, (x, pad),
                        cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), tk, cv2.LINE_AA)
        return cmp

    def _show_preview(self, bgr):
        pil     = cv2_to_photoimage(bgr, max_w=900, max_h=340)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                                size=(pil.width, pil.height))
        self._preview_lbl.configure(image=ctk_img, text="")
        self._preview_lbl._image = ctk_img


# ── Main Application ──────────────────────────────────────────────────────────

class App(ctk.CTk):

    def __init__(self):
        super().__init__()
        self.title("Camera Calibration Tool")
        self.geometry("1020x820")
        self.minsize(900, 700)
        self._webcam_tab = None
        self._build()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        """앱 종료 시 웹캠 스레드를 안전하게 정리."""
        if self._webcam_tab is not None and self._webcam_tab._running:
            self._webcam_tab._stop_flag.set()
            if self._webcam_tab._thread:
                self._webcam_tab._thread.join(timeout=1.0)
        self.destroy()

    def _build(self):
        # Header
        header = ctk.CTkFrame(self, height=56, corner_radius=0, fg_color=BG_CARD)
        header.pack(fill="x")
        header.pack_propagate(False)
        ctk.CTkLabel(header,
                     text="  Camera Calibration & Lens Distortion Correction",
                     font=("Segoe UI", 17, "bold"),
                     text_color="white").pack(side="left", padx=20)
        ctk.CTkLabel(header,
                     text="Webcam · OpenCV",
                     font=("Segoe UI", 12),
                     text_color=TEXT_DIM).pack(side="right", padx=24)

        # Tabs
        tabs = ctk.CTkTabview(self, corner_radius=0)
        tabs.pack(fill="both", expand=True, padx=0, pady=0)

        tabs.add("🎥  Webcam Recorder")
        tabs.add("📐  Calibration")
        tabs.add("✨  Correction")

        # 탭2를 먼저 생성해 탭1에 참조 전달 (녹화 완료 시 경로 자동 입력)
        calib_tab = CalibrationTab(tabs.tab("📐  Calibration"))
        calib_tab.pack(fill="both", expand=True)

        self._webcam_tab = WebcamRecorderTab(
            tabs.tab("🎥  Webcam Recorder"), calib_tab=calib_tab)
        self._webcam_tab.pack(fill="both", expand=True)

        CorrectionTab(tabs.tab("✨  Correction")).pack(fill="both", expand=True)


if __name__ == "__main__":
    App().mainloop()
