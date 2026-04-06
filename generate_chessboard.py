"""
generate_chessboard.py
----------------------
Generate a printable chessboard calibration pattern for camera calibration.
Outputs an A4-sized PNG at 300 DPI suitable for printing.

Usage:
    python generate_chessboard.py [--cols 10] [--rows 7] [--square_mm 25] [--out chessboard.png]

Defaults produce a 10x7 square grid (9x6 inner corners) with 25 mm squares.
"""

import argparse
import numpy as np
import cv2


def generate_chessboard(cols: int, rows: int, square_mm: float, out_path: str, dpi: int = 300):
    """
    Generate and save a chessboard calibration target as a PNG image.

    Parameters
    ----------
    cols : int
        Number of squares in the horizontal direction (inner corners = cols-1).
    rows : int
        Number of squares in the vertical direction  (inner corners = rows-1).
    square_mm : float
        Physical side length of one square in millimetres.
    out_path : str
        Output file path.
    dpi : int
        Dots per inch for the output image.
    """
    # Convert mm to pixels
    px_per_mm = dpi / 25.4
    sq_px = int(round(square_mm * px_per_mm))

    # Board dimensions in pixels
    board_w = cols * sq_px
    board_h = rows * sq_px

    margin_px = int(round(10 * px_per_mm))

    # A4 dimensions — choose portrait or landscape based on board aspect ratio
    a4_short = int(round(210 * px_per_mm))
    a4_long  = int(round(297 * px_per_mm))

    # Try portrait first, then landscape
    if board_w <= a4_short - 2 * margin_px and board_h <= a4_long - 2 * margin_px:
        a4_w, a4_h = a4_short, a4_long     # portrait
        orientation = "Portrait"
    elif board_w <= a4_long - 2 * margin_px and board_h <= a4_short - 2 * margin_px:
        a4_w, a4_h = a4_long, a4_short     # landscape
        orientation = "Landscape"
    else:
        raise ValueError(
            f"Pattern ({board_w}x{board_h} px = "
            f"{cols*square_mm:.0f}x{rows*square_mm:.0f} mm) does not fit on A4 "
            f"in either orientation at {dpi} DPI with 10 mm margins. "
            "Reduce square_mm, cols, or rows."
        )

    max_w = a4_w - 2 * margin_px
    max_h = a4_h - 2 * margin_px
    print(f"[INFO] Orientation: A4 {orientation}")

    # Create white A4 canvas
    canvas = np.ones((a4_h, a4_w), dtype=np.uint8) * 255

    # Draw chessboard centred on canvas
    offset_x = (a4_w - board_w) // 2
    offset_y = (a4_h - board_h) // 2

    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                x0 = offset_x + c * sq_px
                y0 = offset_y + r * sq_px
                canvas[y0:y0 + sq_px, x0:x0 + sq_px] = 0  # black square

    # Draw thin border around the entire board
    cv2.rectangle(
        canvas,
        (offset_x - 1, offset_y - 1),
        (offset_x + board_w, offset_y + board_h),
        color=0,
        thickness=2,
    )

    # Annotate with pattern info (bottom margin)
    inner_cols = cols - 1
    inner_rows = rows - 1
    label = (
        f"Chessboard {inner_cols}x{inner_rows} inner corners | "
        f"Square size: {square_mm:.1f} mm | DPI: {dpi}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.9
    thickness = 2
    text_size, _ = cv2.getTextSize(label, font, font_scale, thickness)
    text_x = (a4_w - text_size[0]) // 2
    text_y = offset_y + board_h + int(round(7 * px_per_mm))
    if text_y + text_size[1] < a4_h:
        cv2.putText(canvas, label, (text_x, text_y), font, font_scale, 0, thickness, cv2.LINE_AA)

    cv2.imwrite(out_path, canvas)
    print(f"[OK] Chessboard saved → {out_path}")
    print(f"     Pattern   : {cols} x {rows} squares  ({inner_cols} x {inner_rows} inner corners)")
    print(f"     Square    : {square_mm:.1f} mm  ({sq_px} px at {dpi} DPI)")
    print(f"     Board size: {board_w} x {board_h} px  ({board_w/px_per_mm:.1f} x {board_h/px_per_mm:.1f} mm)")
    print(f"     Canvas    : A4 ({a4_w} x {a4_h} px at {dpi} DPI)")
    print()
    print("  >> Print at 100% scale (no scaling / fit-to-page).")
    print("  >> Attach the printout flat onto a rigid surface (clipboard, book, etc.).")
    print("  >> Measure the actual printed square size and update --square_mm if needed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a printable chessboard calibration target.")
    parser.add_argument("--cols", type=int, default=10,
                        help="Number of squares horizontally (default: 10 → 9 inner corners)")
    parser.add_argument("--rows", type=int, default=7,
                        help="Number of squares vertically (default: 7 → 6 inner corners)")
    parser.add_argument("--square_mm", type=float, default=25.0,
                        help="Square side length in mm (default: 25.0)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Output resolution in DPI (default: 300)")
    parser.add_argument("--out", type=str, default="chessboard.png",
                        help="Output PNG file path (default: chessboard.png)")
    args = parser.parse_args()

    generate_chessboard(args.cols, args.rows, args.square_mm, args.out, args.dpi)
