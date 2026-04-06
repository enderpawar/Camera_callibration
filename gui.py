"""
gui.py
------
Graphical User Interface for the Camera Calibration & Lens Distortion Correction pipeline.

Usage:
    python gui.py

Tabs:
  1. Chessboard Generator  — create and preview a printable calibration target
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

class ChessboardTab(ctk.CTkFrame):

    def __init__(self, master):
        super().__init__(master, fg_color="transparent")
        self._build()

    def _build(self):
        # ── Controls ──────────────────────────────────────────────────────────
        ctrl = ctk.CTkFrame(self, corner_radius=12)
        ctrl.pack(fill="x", padx=20, pady=(20, 10))

        ctk.CTkLabel(ctrl, text="Chessboard Generator",
                     font=("Segoe UI", 18, "bold")).grid(
            row=0, column=0, columnspan=4, padx=20, pady=(16, 10), sticky="w")

        params = [
            ("Squares (cols)", "10"),
            ("Squares (rows)", "7"),
            ("Square size (mm)", "25"),
            ("DPI", "300"),
        ]
        self._pvars = []
        for i, (lbl, val) in enumerate(params):
            ctk.CTkLabel(ctrl, text=lbl, text_color=TEXT_DIM).grid(
                row=1, column=i * 2, padx=(20 if i == 0 else 10, 4), pady=10, sticky="e")
            var = ctk.StringVar(value=val)
            ctk.CTkEntry(ctrl, textvariable=var, width=75).grid(
                row=1, column=i * 2 + 1, padx=(0, 10), pady=10)
            self._pvars.append(var)

        self._out_var = ctk.StringVar(value="chessboard.png")
        ctk.CTkLabel(ctrl, text="Output file", text_color=TEXT_DIM).grid(
            row=2, column=0, padx=(20, 4), pady=(0, 14), sticky="e")
        ctk.CTkEntry(ctrl, textvariable=self._out_var, width=300).grid(
            row=2, column=1, columnspan=5, padx=(0, 10), pady=(0, 14), sticky="w")

        ctk.CTkButton(ctrl, text="Generate & Preview",
                      command=self._run, width=180).grid(
            row=2, column=6, padx=20, pady=(0, 14))

        # ── Preview ───────────────────────────────────────────────────────────
        prev_frame = ctk.CTkFrame(self, corner_radius=12)
        prev_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        ctk.CTkLabel(prev_frame, text="Preview",
                     font=("Segoe UI", 13, "bold"),
                     text_color=TEXT_DIM).pack(anchor="w", padx=16, pady=(10, 4))

        self._preview_lbl = ctk.CTkLabel(prev_frame, text="Generate a chessboard to see the preview.",
                                         text_color=TEXT_DIM)
        self._preview_lbl.pack(expand=True)

        self._status = ctk.CTkLabel(self, text="", text_color="#aaddaa")
        self._status.pack(pady=(0, 8))

    def _run(self):
        try:
            cols       = int(self._pvars[0].get())
            rows       = int(self._pvars[1].get())
            square_mm  = float(self._pvars[2].get())
            dpi        = int(self._pvars[3].get())
            out_path   = self._out_var.get().strip() or "chessboard.png"
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            return

        try:
            image = self._generate(cols, rows, square_mm, dpi)
            cv2.imwrite(out_path, image)
            self._status.configure(text=f"Saved → {out_path}", text_color="#aaddaa")
            self._show_preview(image)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _generate(self, cols, rows, square_mm, dpi):
        px_per_mm = dpi / 25.4
        sq_px     = int(round(square_mm * px_per_mm))
        board_w   = cols * sq_px
        board_h   = rows * sq_px
        margin_px = int(round(10 * px_per_mm))
        a4_short  = int(round(210 * px_per_mm))
        a4_long   = int(round(297 * px_per_mm))

        if board_w <= a4_short - 2 * margin_px and board_h <= a4_long - 2 * margin_px:
            a4_w, a4_h = a4_short, a4_long
        elif board_w <= a4_long - 2 * margin_px and board_h <= a4_short - 2 * margin_px:
            a4_w, a4_h = a4_long, a4_short
        else:
            raise ValueError(
                f"Pattern ({cols*square_mm:.0f}x{rows*square_mm:.0f} mm) "
                "does not fit on A4. Reduce square_mm or number of squares."
            )

        canvas   = np.ones((a4_h, a4_w), dtype=np.uint8) * 255
        offset_x = (a4_w - board_w) // 2
        offset_y = (a4_h - board_h) // 2

        for r in range(rows):
            for c in range(cols):
                if (r + c) % 2 == 0:
                    x0 = offset_x + c * sq_px
                    y0 = offset_y + r * sq_px
                    canvas[y0:y0 + sq_px, x0:x0 + sq_px] = 0

        cv2.rectangle(canvas, (offset_x - 1, offset_y - 1),
                      (offset_x + board_w, offset_y + board_h), 0, 2)

        label = (f"Chessboard {cols-1}x{rows-1} inner corners | "
                 f"Square: {square_mm:.1f} mm | DPI: {dpi}")
        font_scale = 0.9
        thickness  = 2
        font       = cv2.FONT_HERSHEY_SIMPLEX
        tw, th     = cv2.getTextSize(label, font, font_scale, thickness)[0]
        tx = (a4_w - tw) // 2
        ty = offset_y + board_h + int(round(7 * px_per_mm))
        if ty + th < a4_h:
            cv2.putText(canvas, label, (tx, ty), font, font_scale, 0, thickness, cv2.LINE_AA)

        return canvas

    def _show_preview(self, gray_image):
        # Downscale for display
        bgr = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
        pil = cv2_to_photoimage(bgr, max_w=860, max_h=480)
        ctk_img = ctk.CTkImage(light_image=pil, dark_image=pil,
                                size=(pil.width, pil.height))
        self._preview_lbl.configure(image=ctk_img, text="")
        self._preview_lbl._image = ctk_img  # prevent GC


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
            row=2, column=0, columnspan=3, padx=0, pady=(4, 0), sticky="w")

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
        self._build()

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
                     text="iPhone 15 Pro · OpenCV",
                     font=("Segoe UI", 12),
                     text_color=TEXT_DIM).pack(side="right", padx=24)

        # Tabs
        tabs = ctk.CTkTabview(self, corner_radius=0)
        tabs.pack(fill="both", expand=True, padx=0, pady=0)

        tabs.add("🖨  Chessboard")
        tabs.add("📐  Calibration")
        tabs.add("✨  Correction")

        ChessboardTab(tabs.tab("🖨  Chessboard")).pack(fill="both", expand=True)
        CalibrationTab(tabs.tab("📐  Calibration")).pack(fill="both", expand=True)
        CorrectionTab(tabs.tab("✨  Correction")).pack(fill="both", expand=True)


if __name__ == "__main__":
    App().mainloop()
