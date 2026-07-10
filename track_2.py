import cv2
import numpy as np
import csv
import os

# ── USER CONFIG ──────────────────────────────────────────────────────────────
VIDEO_PATH = r"C:\Users\soumy\intern_py\vid_8.MOV"
TEMPLATE_PATH = "marker_template.png"

OUTPUT_VIDEO = "tracked_bullseye_fixed.mp4"
OUTPUT_CSV = "tracked_height_data.csv"
VELOCITY_CSV = "vertical_velocity.csv"

# Motion reference requested by you
THETA_START_DEG = 38.0
THETA_END_DEG = 90.0

# Physical height should start from 0 and increase with time.
# For normal image coordinates, y increases downward.
# Use "down" if plunge/downward motion should be positive height.
# Use "up" only if your camera is flipped relative to the mechanism motion.
HEIGHT_POSITIVE_DIR = "down"

# Calibration:
# Use ONE physical marker dimension consistently.
# Primary: outer circular bullseye diameter if structural circle is detected.
OUTER_CIRCLE_DIAMETER_MM = 11.0

# Fallback: width of the reference template if full-frame template match is used.
REFERENCE_TEMPLATE_WIDTH_MM = 13.0

# Optional reference model columns for comparison with your theory curve
EXPORT_REFERENCE_MODEL = True
MODEL_C0_MM = 20.0
MODEL_R_MM = 25.0
MODEL_L_MM = 75.0

# ── TEMPLATE MATCHING (primary per-frame detector) ───────────────────────────
MATCH_THRESH = 0.30
SEARCH_MARGIN = 150
ROI_MATCH_SCALES = [0.85, 1.0, 1.15]

# ── RE-DETECTION TRIGGER ─────────────────────────────────────────────────────
LOST_THRESH = 6

# ── STRUCTURAL DETECTION (init + re-detect only) ─────────────────────────────
MIN_CIRCULARITY = 0.55
MIN_RADIUS_PX = 6
MAX_RADIUS_FRAC = 0.30
CONCENTRIC_TOL = 14
MIN_RINGS = 2
TMPL_SCALES = np.linspace(0.3, 2.0, 25)
TMPL_MIN_SCORE = 0.30

# ── KALMAN ────────────────────────────────────────────────────────────────────
KF_PROC_NOISE = 10.0
KF_MEAS_NOISE = 1.5

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ STRUCTURAL DETECTION
# ╚══════════════════════════════════════════════════════════════════════════════

def _circularity(cnt):
    a = cv2.contourArea(cnt)
    p = cv2.arcLength(cnt, True)
    return (4 * np.pi * a / (p * p)) if p > 1e-6 else 0.0

def _circle_candidates(img_bgr, max_r_px):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if img_bgr.ndim == 3 else img_bgr
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kern = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kern)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kern)
    cnts, _ = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

    out = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < np.pi * MIN_RADIUS_PX ** 2 or len(c) < 10:
            continue
        if _circularity(c) < MIN_CIRCULARITY:
            continue
        M = cv2.moments(c)
        if M["m00"] < 1:
            continue
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        r = np.sqrt(area / np.pi)
        if r <= max_r_px:
            out.append((cx, cy, r))
    return out

def detect_structural(img_bgr):
    h, w = img_bgr.shape[:2]
    cands = _circle_candidates(img_bgr, MAX_RADIUS_FRAC * min(h, w))
    if len(cands) < MIN_RINGS:
        return None, None, None, 0.0

    best = []
    for cx0, cy0, _ in cands:
        grp = [(cx, cy, r) for cx, cy, r in cands if np.hypot(cx - cx0, cy - cy0) < CONCENTRIC_TOL]
        if len(grp) > len(best):
            best = grp

    if len(best) < MIN_RINGS:
        return None, None, None, 0.0

    cx_f = float(np.mean([g[0] for g in best]))
    cy_f = float(np.mean([g[1] for g in best]))
    r_out = float(max(g[2] for g in best))
    conf = min(1.0, len(best) / 3.0)
    return cx_f, cy_f, r_out, conf

def template_fullframe(frame_bgr, tmpl_gray):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    fh, fw = gray.shape
    th, tw = tmpl_gray.shape

    best = {
        "cx": None, "cy": None, "half": None, "score": -1.0,
        "w": None, "h": None
    }

    for s in TMPL_SCALES:
        nw, nh = int(tw * s), int(th * s)
        if nw < 15 or nh < 15 or nw >= fw or nh >= fh:
            continue

        resized = cv2.resize(tmpl_gray, (nw, nh))
        res = cv2.matchTemplate(gray, resized, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv2.minMaxLoc(res)

        if maxv > best["score"]:
            best.update({
                "cx": maxl[0] + nw // 2,
                "cy": maxl[1] + nh // 2,
                "half": max(nw, nh) // 2,
                "score": maxv,
                "w": nw,
                "h": nh
            })

    if best["score"] < TMPL_MIN_SCORE:
        return None, None, None, 0.0, None, None

    return best["cx"], best["cy"], best["half"], best["score"], best["w"], best["h"]

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ LOCAL TEMPLATE EXTRACTION
# ╚══════════════════════════════════════════════════════════════════════════════

def extract_local_template(frame_bgr, cx, cy, radius):
    h, w = frame_bgr.shape[:2]
    half = max(int(radius * 1.4), 40)
    half = min(half, 200)

    x1 = max(0, int(cx) - half)
    y1 = max(0, int(cy) - half)
    x2 = min(w, int(cx) + half)
    y2 = min(h, int(cy) + half)

    patch = frame_bgr[y1:y2, x1:x2]
    return cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY), (x1, y1)

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ ROI TEMPLATE MATCHING
# ╚══════════════════════════════════════════════════════════════════════════════

def match_in_roi(roi_bgr, local_tmpl_gray):
    roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
    rh, rw = roi_gray.shape
    th, tw = local_tmpl_gray.shape
    best = (None, None, -1.0)

    for s in ROI_MATCH_SCALES:
        nw, nh = int(tw * s), int(th * s)
        if nw < 10 or nh < 10 or nw >= rw or nh >= rh:
            continue

        tmpl_s = cv2.resize(local_tmpl_gray, (nw, nh))
        res = cv2.matchTemplate(roi_gray, tmpl_s, cv2.TM_CCOEFF_NORMED)
        _, maxv, _, maxl = cv2.minMaxLoc(res)

        if maxv > best[2]:
            best = (
                float(maxl[0] + nw // 2),
                float(maxl[1] + nh // 2),
                float(maxv)
            )

    if best[2] < MATCH_THRESH:
        return None, None, best[2]

    return best[0], best[1], best[2]

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ KALMAN
# ╚══════════════════════════════════════════════════════════════════════════════

def make_kalman(cx0, cy0, fps):
    dt = 1.0 / fps
    kf = cv2.KalmanFilter(4, 2)

    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ], dtype=np.float32)

    kf.measurementMatrix = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ], dtype=np.float32)

    kf.processNoiseCov = np.eye(4, dtype=np.float32) * KF_PROC_NOISE
    kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * KF_MEAS_NOISE
    kf.errorCovPost = np.eye(4, dtype=np.float32)
    kf.statePost = np.array([[cx0], [cy0], [0.0], [0.0]], dtype=np.float32)
    return kf

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ THEORY / EXPORT HELPERS
# ╚══════════════════════════════════════════════════════════════════════════════

def reference_theta_deg(frame_idx, total_frames, theta_start_deg, theta_end_deg):
    if total_frames <= 1:
        return theta_start_deg
    alpha = frame_idx / (total_frames - 1)
    return theta_start_deg + alpha * (theta_end_deg - theta_start_deg)

def exact_slider_displacement_mm(theta_rad, c0_mm, r_mm, l_mm):
    under_root = l_mm**2 - (r_mm * np.sin(theta_rad))**2
    under_root = max(under_root, 1e-9)
    return c0_mm - r_mm * np.cos(theta_rad) + np.sqrt(under_root) - l_mm + r_mm

def compute_height_mm(cy_now, cy_init, mm_per_px):
    if HEIGHT_POSITIVE_DIR.lower() == "down":
        h = (cy_now - cy_init) * mm_per_px
    else:
        h = (cy_init - cy_now) * mm_per_px
    return max(0.0, float(h))

def estimate_vertical_velocity(records):
    if len(records) < 2:
        return None, None, None

    times = np.array([r["time_s"] for r in records], dtype=float)
    heights_mm = np.array([r["height_mm"] for r in records], dtype=float)

    vel_mm_s = np.gradient(heights_mm, times)
    vel_m_s = vel_mm_s / 1000.0
    return times, heights_mm, vel_m_s

# ╔══════════════════════════════════════════════════════════════════════════════
# ║ MAIN
# ╚══════════════════════════════════════════════════════════════════════════════

def main():
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(VIDEO_PATH)

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[INFO] {W}x{H}, {fps:.3f} fps, {total} frames")

    ref_tmpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if ref_tmpl is None:
        raise FileNotFoundError(TEMPLATE_PATH)

    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Cannot read frame 0.")

    cx0, cy0, r0, c0 = detect_structural(frame0)

    mm_per_px = None
    init_radius_for_template = None

    if cx0 is not None and r0 is not None and r0 > 0:
        mm_per_px = OUTER_CIRCLE_DIAMETER_MM / (2.0 * r0)
        init_radius_for_template = r0
        print(f"[INIT] Structural detection: ({cx0:.2f}, {cy0:.2f}), r={r0:.2f}px, conf={c0:.2f}")
        print(f"[CAL] Using outer-circle diameter = {OUTER_CIRCLE_DIAMETER_MM:.3f} mm")
        print(f"[CAL] Scale = {mm_per_px:.6f} mm/px")
    else:
        print("[INIT] Structural detection failed; trying full-frame reference template match...")
        cx0, cy0, half0, c0, match_w0, match_h0 = template_fullframe(frame0, ref_tmpl)

        if cx0 is None:
            raise RuntimeError(
                "Marker not found in frame 0.\n"
                "Check TEMPLATE_PATH and ensure the marker is visible in the first frame."
            )

        mm_per_px = REFERENCE_TEMPLATE_WIDTH_MM / float(match_w0)
        init_radius_for_template = max(match_w0, match_h0) / 2.0
        print(f"[INIT] Template detection: ({cx0:.2f}, {cy0:.2f}), score={c0:.2f}")
        print(f"[CAL] Using reference template width = {REFERENCE_TEMPLATE_WIDTH_MM:.3f} mm")
        print(f"[CAL] Matched template width = {match_w0}px")
        print(f"[CAL] Scale = {mm_per_px:.6f} mm/px")

    local_tmpl, _ = extract_local_template(frame0, cx0, cy0, init_radius_for_template)
    tmpl_h, tmpl_w = local_tmpl.shape
    half_sw = max(tmpl_w // 2, tmpl_h // 2, 50)

    kf = make_kalman(float(cx0), float(cy0), fps)
    lost_count = 0
    initial_cy = float(cy0)

    writer = cv2.VideoWriter(
        OUTPUT_VIDEO,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (W, H)
    )

    model_ref0 = exact_slider_displacement_mm(
        np.deg2rad(THETA_START_DEG), MODEL_C0_MM, MODEL_R_MM, MODEL_L_MM
    ) if EXPORT_REFERENCE_MODEL else None

    csv_header = [
        "frame",
        "time_s",
        "theta_ref_deg",
        "cx_px",
        "cy_px",
        "height_mm",
        "confidence",
        "detected"
    ]
    if EXPORT_REFERENCE_MODEL:
        csv_header.append("slider_ref_rel_mm")

    records = []

    with open(OUTPUT_CSV, "w", newline="") as fpos:
        wpos = csv.writer(fpos)
        wpos.writerow(csv_header)

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        for frame_idx in range(total):
            ok, frame = cap.read()
            if not ok:
                break

            t_s = frame_idx / fps
            theta_ref_deg = reference_theta_deg(frame_idx, total, THETA_START_DEG, THETA_END_DEG)
            theta_ref_rad = np.deg2rad(theta_ref_deg)

            slider_ref_rel_mm = None
            if EXPORT_REFERENCE_MODEL:
                slider_ref_now = exact_slider_displacement_mm(
                    theta_ref_rad, MODEL_C0_MM, MODEL_R_MM, MODEL_L_MM
                )
                slider_ref_rel_mm = slider_ref_now - model_ref0

            disp = frame.copy()

            pred = kf.predict().flatten()
            px, py = float(pred[0]), float(pred[1])

            x1 = max(0, int(px) - half_sw - SEARCH_MARGIN)
            y1 = max(0, int(py) - half_sw - SEARCH_MARGIN)
            x2 = min(W, int(px) + half_sw + SEARCH_MARGIN)
            y2 = min(H, int(py) + half_sw + SEARCH_MARGIN)

            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 140, 0), 2)

            cx_l, cy_l, score = match_in_roi(frame[y1:y2, x1:x2], local_tmpl)

            detected = False
            confidence = 0.0
            cx_use, cy_use = px, py

            if cx_l is None and lost_count >= LOST_THRESH:
                cx_fd, cy_fd, r_fd, c_fd = detect_structural(frame)

                if cx_fd is None:
                    cx_fd, cy_fd, half_fd, c_fd, mw_fd, mh_fd = template_fullframe(frame, ref_tmpl)
                    if cx_fd is not None:
                        r_fd = max(mw_fd, mh_fd) / 2.0

                if cx_fd is not None:
                    kf = make_kalman(float(cx_fd), float(cy_fd), fps)
                    local_tmpl, _ = extract_local_template(frame, cx_fd, cy_fd, r_fd)
                    meas = np.array([[cx_fd], [cy_fd]], dtype=np.float32)
                    kf.correct(meas)
                    state = kf.statePost.flatten()

                    cx_use = float(state[0])
                    cy_use = float(state[1])
                    detected = True
                    confidence = float(c_fd)
                    lost_count = 0
                else:
                    lost_count += 1
            elif cx_l is not None:
                cx_g = x1 + cx_l
                cy_g = y1 + cy_l
                meas = np.array([[cx_g], [cy_g]], dtype=np.float32)
                kf.correct(meas)
                state = kf.statePost.flatten()

                cx_use = float(state[0])
                cy_use = float(state[1])
                detected = True
                confidence = float(score)
                lost_count = 0
            else:
                lost_count += 1

            height_mm = compute_height_mm(cy_use, initial_cy, mm_per_px)

            ix, iy = int(round(cx_use)), int(round(cy_use))
            if detected:
                cv2.circle(disp, (ix, iy), 8, (0, 0, 255), -1)
                cv2.line(disp, (ix - 22, iy), (ix + 22, iy), (0, 255, 0), 2)
                cv2.line(disp, (ix, iy - 22), (ix, iy + 22), (0, 255, 0), 2)
                cv2.putText(disp, f"det score={confidence:.2f}", (12, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 0), 2)
            else:
                cv2.circle(disp, (ix, iy), 8, (80, 80, 255), -1)
                cv2.putText(disp, f"coasting [{lost_count}]", (12, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (80, 80, 255), 2)

            cv2.putText(disp, f"height = {height_mm:.3f} mm", (12, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
            cv2.putText(disp, f"theta_ref = {theta_ref_deg:.2f} deg", (12, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
            cv2.putText(disp, f"scale = {mm_per_px:.5f} mm/px", (12, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 200, 0), 2)

            row = [
                frame_idx,
                t_s,
                theta_ref_deg,
                cx_use,
                cy_use,
                height_mm,
                confidence,
                int(detected)
            ]
            if EXPORT_REFERENCE_MODEL:
                row.append(slider_ref_rel_mm)

            wpos.writerow(row)

            records.append({
                "frame": frame_idx,
                "time_s": t_s,
                "theta_ref_deg": theta_ref_deg,
                "cx_px": cx_use,
                "cy_px": cy_use,
                "height_mm": height_mm,
                "confidence": confidence,
                "detected": int(detected),
                "slider_ref_rel_mm": slider_ref_rel_mm
            })

            writer.write(disp)

    writer.release()
    cap.release()

    print(f"[DONE] Saved video: {OUTPUT_VIDEO}")
    print(f"[DONE] Saved height data: {OUTPUT_CSV}")

    vel_out = estimate_vertical_velocity(records)
    if vel_out is not None:
        times, heights_mm, vel_m_s = vel_out

        with open(VELOCITY_CSV, "w", newline="") as fv:
            wv = csv.writer(fv)
            header = ["time_s", "theta_ref_deg", "height_mm", "vertical_velocity_m_per_s", "detected"]
            if EXPORT_REFERENCE_MODEL:
                header.append("slider_ref_rel_mm")
            wv.writerow(header)

            for rec, t, h, v in zip(records, times, heights_mm, vel_m_s):
                row = [t, rec["theta_ref_deg"], h, v, rec["detected"]]
                if EXPORT_REFERENCE_MODEL:
                    row.append(rec["slider_ref_rel_mm"])
                wv.writerow(row)

        print(f"[DONE] Saved vertical velocity: {VELOCITY_CSV}")
        print(f"[INFO] Peak vertical velocity = {np.max(vel_m_s):.6f} m/s")
        print(f"[INFO] Mean vertical velocity = {np.mean(vel_m_s):.6f} m/s")

if __name__ == "__main__":
    main()