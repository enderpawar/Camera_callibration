"""
camera_calibration.py
---------------------
Perform camera calibration from a chessboard video (or image folder).

Usage:
    python camera_calibration.py --video data/chessboard.mp4
    python camera_calibration.py --video data/chessboard.mp4 --cols 9 --rows 6 --square_mm 25
    python camera_calibration.py --video data/chessboard.mp4 --step 10 --min_frames 20

Arguments:
    --video       Path to the input chessboard video (required)
    --cols        Number of inner corners horizontally (default: 9)
    --rows        Number of inner corners vertically   (default: 6)
    --square_mm   Physical square size in mm           (default: 25.0)
    --step        Process every N-th frame             (default: 5)
    --min_frames  Minimum frames required for calibration (default: 20)
    --save_frames Save detected-corner frames to results/calibration_frames/
    --output      Path to save calibration result .npz (default: results/calibration.npz)
    --no_display  Disable live preview window
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_corners(gray: np.ndarray, pattern_size: tuple) -> tuple[bool, np.ndarray | None]:
    """
    Detect chessboard corners with subpixel refinement.

    Returns (found, corners_refined).
    """
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)

    if not found:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners_refined


def build_object_points(pattern_size: tuple, square_mm: float) -> np.ndarray:
    """
    Build the 3-D world coordinates of the chessboard corners.
    The board lies on the Z=0 plane.
    """
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_mm
    return objp


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def calibrate(args):
    pattern_size = (args.cols, args.rows)   # (inner cols, inner rows)
    objp = build_object_points(pattern_size, args.square_mm)

    obj_points = []   # 3-D world points per frame
    img_points = []   # 2-D image points per frame

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps          = cap.get(cv2.CAP_PROP_FPS)
    frame_w      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_size     = (frame_w, frame_h)

    print(f"[INFO] Video : {args.video}")
    print(f"       Size  : {frame_w} x {frame_h}  |  FPS: {fps:.2f}  |  Frames: {total_frames}")
    print(f"[INFO] Pattern: {args.cols} x {args.rows} inner corners | Square: {args.square_mm} mm")
    print(f"[INFO] Sampling every {args.step} frames (target ≥ {args.min_frames} detections)")
    print()

    # Output directories
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    frames_dir = None
    if args.save_frames:
        frames_dir = os.path.join(os.path.dirname(args.output), "calibration_frames")
        os.makedirs(frames_dir, exist_ok=True)

    frame_idx   = 0
    detect_cnt  = 0
    saved_cnt   = 0
    t_start     = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % args.step != 0:
            frame_idx += 1
            continue

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = find_corners(gray, pattern_size)

        if found:
            obj_points.append(objp)
            img_points.append(corners)
            detect_cnt += 1

            # Draw corners on a copy for display / saving
            vis = frame.copy()
            cv2.drawChessboardCorners(vis, pattern_size, corners, found)

            status_text = f"Detected: {detect_cnt} / min {args.min_frames}  (frame {frame_idx})"
            cv2.putText(vis, status_text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

            if frames_dir and saved_cnt < 50:          # cap at 50 saved frames
                cv2.imwrite(os.path.join(frames_dir, f"frame_{frame_idx:05d}.jpg"), vis)
                saved_cnt += 1

            if not args.no_display:
                cv2.imshow("Calibration — corner detection (press Q to stop early)", vis)
        else:
            if not args.no_display:
                info = frame.copy()
                cv2.putText(info, f"Searching...  detected so far: {detect_cnt}",
                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 140, 255), 2, cv2.LINE_AA)
                cv2.imshow("Calibration — corner detection (press Q to stop early)", info)

        if not args.no_display:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Stopped early by user.")
                break

        frame_idx += 1

    cap.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    elapsed = time.time() - t_start
    print(f"[INFO] Frame scan complete in {elapsed:.1f}s — {detect_cnt} frames with detected corners")

    if detect_cnt < args.min_frames:
        sys.exit(
            f"[ERROR] Only {detect_cnt} frames detected (need ≥ {args.min_frames}). "
            "Record more diverse viewpoints and try again."
        )

    # -----------------------------------------------------------------------
    # Camera calibration
    # -----------------------------------------------------------------------
    print(f"\n[INFO] Running cv2.calibrateCamera() on {detect_cnt} frames …")

    # CALIB_FIX_K3: k3를 0으로 고정 → 고차 방사형 왜곡 계수가 노이즈를 과적합하는 것을 방지
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, img_size, None, None,
        flags=cv2.CALIB_FIX_K3
    )

    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    # dist_coeffs shape: (1, 5) → [k1, k2, p1, p2, k3]
    d = dist_coeffs.ravel()
    k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
    k3 = d[4] if len(d) > 4 else 0.0

    # Per-view reprojection errors
    per_view_errors = []
    for i in range(len(obj_points)):
        proj, _ = cv2.projectPoints(obj_points[i], rvecs[i], tvecs[i],
                                    camera_matrix, dist_coeffs)
        err = cv2.norm(img_points[i], proj, cv2.NORM_L2) / len(proj)
        per_view_errors.append(err)

    print()
    print("=" * 55)
    print("  Camera Calibration Results")
    print("=" * 55)
    print(f"  Camera        : {args.video}")
    print(f"  Image size    : {frame_w} x {frame_h} px")
    print(f"  Frames used   : {detect_cnt}")
    print("-" * 55)
    print(f"  fx            : {fx:.4f} px")
    print(f"  fy            : {fy:.4f} px")
    print(f"  cx            : {cx:.4f} px")
    print(f"  cy            : {cy:.4f} px")
    print("-" * 55)
    print(f"  k1            : {k1:.6f}")
    print(f"  k2            : {k2:.6f}")
    print(f"  p1            : {p1:.6f}")
    print(f"  p2            : {p2:.6f}")
    print(f"  k3            : {k3:.6f}")
    print("-" * 55)
    print(f"  RMSE          : {rms:.4f} px")
    print(f"  Max view err  : {max(per_view_errors):.4f} px")
    print(f"  Min view err  : {min(per_view_errors):.4f} px")
    print("=" * 55)

    if rms < 0.5:
        print("  [Excellent] RMSE < 0.5 px — very high quality calibration.")
    elif rms < 1.0:
        print("  [Good] RMSE < 1.0 px — acceptable for most applications.")
    else:
        print("  [Warning] RMSE ≥ 1.0 px — consider recapturing with more diverse viewpoints.")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    np.savez(
        args.output,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        img_size=np.array(img_size),
        rms=np.array(rms),
        square_mm=np.array(args.square_mm),
    )
    print(f"\n[OK] Calibration data saved → {args.output}")

    if frames_dir:
        print(f"[OK] Sample frames saved  → {frames_dir}/  ({saved_cnt} files)")

    return camera_matrix, dist_coeffs, rms


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Camera calibration from chessboard video.")
    parser.add_argument("--video",      required=True,         help="Path to chessboard video file")
    parser.add_argument("--cols",       type=int,   default=9, help="Inner corners horizontally (default: 9)")
    parser.add_argument("--rows",       type=int,   default=6, help="Inner corners vertically   (default: 6)")
    parser.add_argument("--square_mm",  type=float, default=25.0, help="Square size in mm (default: 25.0)")
    parser.add_argument("--step",       type=int,   default=5,  help="Sample every N-th frame   (default: 5)")
    parser.add_argument("--min_frames", type=int,   default=20, help="Min frames needed          (default: 20)")
    parser.add_argument("--save_frames", action="store_true",   help="Save detected-corner frames to results/calibration_frames/")
    parser.add_argument("--output",     default="results/calibration.npz", help="Output .npz path")
    parser.add_argument("--no_display", action="store_true",    help="Disable live preview window")
    args = parser.parse_args()

    calibrate(args)
