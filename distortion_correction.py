"""
distortion_correction.py
-------------------------
Apply lens distortion correction using pre-computed calibration results.

Usage:
    # Correct a video and display/save side-by-side comparison
    python distortion_correction.py --video data/chessboard.mp4

    # Correct a single image
    python distortion_correction.py --image path/to/image.jpg

    # Keep all pixels (alpha=1) instead of cropping black borders (alpha=0)
    python distortion_correction.py --video data/chessboard.mp4 --alpha 1

Arguments:
    --video        Input video file (mutually exclusive with --image)
    --image        Input image file (mutually exclusive with --video)
    --calib        Path to calibration .npz file  (default: results/calibration.npz)
    --alpha        Undistortion alpha 0=crop, 1=keep all  (default: 0)
    --out_video    Output video path (default: results/undistorted.mp4)
    --out_compare  Output comparison JPEG path (default: results/comparison.jpg)
    --no_display   Disable live preview window
    --compare_frame  Frame index to use for the comparison snapshot (default: middle frame)
"""

import argparse
import os
import sys

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Core undistortion helpers
# ---------------------------------------------------------------------------

def load_calibration(calib_path: str):
    """Load camera matrix and distortion coefficients from .npz file."""
    if not os.path.isfile(calib_path):
        sys.exit(f"[ERROR] Calibration file not found: {calib_path}\n"
                 "Run camera_calibration.py first.")
    data = np.load(calib_path)
    camera_matrix = data["camera_matrix"]
    dist_coeffs   = data["dist_coeffs"]
    img_size      = tuple(data["img_size"])   # (W, H)
    rms           = float(data["rms"])
    return camera_matrix, dist_coeffs, img_size, rms


def build_undistort_maps(camera_matrix, dist_coeffs, img_size, alpha: float):
    """
    Pre-compute undistortion / rectification maps for fast per-frame remapping.

    alpha=0 → crop to valid pixels (no black borders)
    alpha=1 → keep all pixels (black borders may appear)
    """
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, alpha, img_size
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix, dist_coeffs, None, new_camera_matrix, img_size,
        cv2.CV_16SC2
    )
    return map1, map2, new_camera_matrix, roi


def undistort_frame(frame: np.ndarray, map1, map2) -> np.ndarray:
    """Apply pre-computed undistortion maps to a single frame."""
    return cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT)


def make_comparison(original: np.ndarray, corrected: np.ndarray,
                    label_orig: str = "Original (distorted)",
                    label_corr: str = "Undistorted") -> np.ndarray:
    """
    Create a side-by-side comparison image with labels.
    Both images are resized to the same height if they differ.
    """
    h = max(original.shape[0], corrected.shape[0])
    # Resize if needed (keep aspect ratio)
    def resize_to_height(img, target_h):
        scale = target_h / img.shape[0]
        return cv2.resize(img, (int(img.shape[1] * scale), target_h),
                          interpolation=cv2.INTER_AREA)

    left  = resize_to_height(original,  h)
    right = resize_to_height(corrected, h)

    # Draw dividing line on right border of left panel
    separator = np.full((h, 4, 3), 128, dtype=np.uint8)
    comparison = np.hstack([left, separator, right])

    # Labels
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.6, h / 1080)
    thickness  = max(1, int(h / 540))
    pad        = int(h * 0.03)

    for text, x_offset in [(label_orig, pad), (label_corr, left.shape[1] + 4 + pad)]:
        # Shadow
        cv2.putText(comparison, text, (x_offset + 1, pad + 1),
                    font, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
        # Text
        cv2.putText(comparison, text, (x_offset, pad),
                    font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

    return comparison


# ---------------------------------------------------------------------------
# Video processing
# ---------------------------------------------------------------------------

def process_video(args, camera_matrix, dist_coeffs, img_size, map1, map2):
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"[ERROR] Cannot open video: {args.video}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    vw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    vh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"[INFO] Video : {args.video}")
    print(f"       Size  : {vw} x {vh}  |  FPS: {fps:.2f}  |  Frames: {total}")

    # Output writer — side-by-side comparison video
    os.makedirs(os.path.dirname(args.out_video) or ".", exist_ok=True)
    compare_w = vw * 2 + 4    # separator width
    compare_h = vh
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out_video, fourcc, fps, (compare_w, compare_h))
    if not writer.isOpened():
        print("[Warning] Could not open VideoWriter — output video will not be saved.")
        writer = None

    compare_frame_idx = args.compare_frame if args.compare_frame >= 0 else total // 2
    snapshot_saved = False

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        corrected   = undistort_frame(frame, map1, map2)
        comparison  = make_comparison(frame, corrected)

        if writer:
            # Ensure the comparison matches the writer's expected size
            if comparison.shape[1] != compare_w or comparison.shape[0] != compare_h:
                comparison_resized = cv2.resize(comparison, (compare_w, compare_h))
            else:
                comparison_resized = comparison
            writer.write(comparison_resized)

        # Save snapshot for README
        if frame_idx >= compare_frame_idx and not snapshot_saved:
            os.makedirs(os.path.dirname(args.out_compare) or ".", exist_ok=True)
            cv2.imwrite(args.out_compare, comparison)
            snapshot_saved = True
            print(f"[OK] Comparison snapshot saved → {args.out_compare}  (frame {frame_idx})")

        if not args.no_display:
            # Scale down for display if too wide
            disp = comparison
            max_disp_w = 1600
            if disp.shape[1] > max_disp_w:
                scale = max_disp_w / disp.shape[1]
                disp  = cv2.resize(disp, (max_disp_w, int(disp.shape[0] * scale)))
            cv2.imshow("Lens Distortion Correction (press Q to quit)", disp)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                print("[INFO] Stopped early by user.")
                break

        frame_idx += 1

        if frame_idx % 100 == 0:
            pct = frame_idx / max(total, 1) * 100
            print(f"  Processing… {frame_idx}/{total} ({pct:.1f}%)")

    cap.release()
    if writer:
        writer.release()
    if not args.no_display:
        cv2.destroyAllWindows()

    if writer:
        print(f"[OK] Corrected comparison video saved → {args.out_video}")

    # Save a snapshot of the last frame if none was captured yet
    if not snapshot_saved and frame_idx > 0:
        print(f"[Warning] compare_frame index ({compare_frame_idx}) not reached; snapshot not saved.")


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------

def process_image(args, camera_matrix, dist_coeffs, img_size, map1, map2):
    frame = cv2.imread(args.image)
    if frame is None:
        sys.exit(f"[ERROR] Cannot read image: {args.image}")

    print(f"[INFO] Image : {args.image}  ({frame.shape[1]} x {frame.shape[0]})")

    corrected  = undistort_frame(frame, map1, map2)
    comparison = make_comparison(frame, corrected)

    os.makedirs(os.path.dirname(args.out_compare) or ".", exist_ok=True)
    cv2.imwrite(args.out_compare, comparison)
    print(f"[OK] Comparison image saved → {args.out_compare}")

    # Save undistorted image
    out_img = args.out_compare.replace("comparison", "undistorted_image")
    cv2.imwrite(out_img, corrected)
    print(f"[OK] Undistorted image saved → {out_img}")

    if not args.no_display:
        disp = comparison
        max_disp_w = 1600
        if disp.shape[1] > max_disp_w:
            scale = max_disp_w / disp.shape[1]
            disp  = cv2.resize(disp, (max_disp_w, int(disp.shape[0] * scale)))
        cv2.imshow("Lens Distortion Correction (press any key to close)", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Lens distortion correction using calibration results.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  help="Input video file")
    group.add_argument("--image",  help="Input image file")

    parser.add_argument("--calib",         default="results/calibration.npz",
                        help="Calibration .npz file (default: results/calibration.npz)")
    parser.add_argument("--alpha",         type=float, default=0,
                        help="Undistortion alpha: 0=crop black borders, 1=keep all pixels (default: 0)")
    parser.add_argument("--out_video",     default="results/undistorted.mp4",
                        help="Output corrected video (default: results/undistorted.mp4)")
    parser.add_argument("--out_compare",   default="results/comparison.jpg",
                        help="Output comparison snapshot (default: results/comparison.jpg)")
    parser.add_argument("--compare_frame", type=int, default=-1,
                        help="Frame index for comparison snapshot, -1 = middle (default: -1)")
    parser.add_argument("--no_display",    action="store_true",
                        help="Disable live preview window")
    args = parser.parse_args()

    # Load calibration
    camera_matrix, dist_coeffs, calib_img_size, rms = load_calibration(args.calib)

    d = dist_coeffs.ravel()
    k1, k2, p1, p2 = d[0], d[1], d[2], d[3]
    k3 = d[4] if len(d) > 4 else 0.0
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]

    print("=" * 55)
    print("  Loaded Calibration Parameters")
    print("=" * 55)
    print(f"  fx={fx:.2f}, fy={fy:.2f}, cx={cx:.2f}, cy={cy:.2f}")
    print(f"  k1={k1:.6f}, k2={k2:.6f}, p1={p1:.6f}, p2={p2:.6f}, k3={k3:.6f}")
    print(f"  Calibration RMSE: {rms:.4f} px")
    print(f"  alpha={args.alpha}  ({'crop black borders' if args.alpha == 0 else 'keep all pixels'})")
    print("=" * 55)
    print()

    # Determine actual image/video size to build maps
    if args.video:
        cap    = cv2.VideoCapture(args.video)
        vw     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vh     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        img_size = (vw, vh)
    else:
        img = cv2.imread(args.image)
        if img is None:
            sys.exit(f"[ERROR] Cannot read image: {args.image}")
        img_size = (img.shape[1], img.shape[0])

    if img_size != calib_img_size:
        print(f"[Warning] Input size ({img_size}) differs from calibration size ({calib_img_size}).")
        print("          Results may be inaccurate. Re-run calibration on matching resolution.")

    # Build undistortion maps once
    map1, map2, new_cam_mat, roi = build_undistort_maps(
        camera_matrix, dist_coeffs, img_size, args.alpha
    )
    print(f"[INFO] New camera matrix (alpha={args.alpha}):")
    print(f"       fx'={new_cam_mat[0,0]:.2f}, fy'={new_cam_mat[1,1]:.2f}, "
          f"cx'={new_cam_mat[0,2]:.2f}, cy'={new_cam_mat[1,2]:.2f}")
    if args.alpha == 0:
        print(f"       Valid ROI: x={roi[0]}, y={roi[1]}, w={roi[2]}, h={roi[3]}")
    print()

    if args.video:
        process_video(args, camera_matrix, dist_coeffs, img_size, map1, map2)
    else:
        process_image(args, camera_matrix, dist_coeffs, img_size, map1, map2)


if __name__ == "__main__":
    main()
