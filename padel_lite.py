import argparse
import json
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


FONT = cv2.FONT_HERSHEY_DUPLEX


def color_for_id(track_id: int) -> tuple[int, int, int]:
    palette = [
        (80, 180, 255),
        (80, 255, 170),
        (255, 160, 80),
        (190, 120, 255),
        (255, 210, 90),
        (90, 200, 255),
    ]
    try:
        idx = int(track_id) % len(palette)
    except (TypeError, ValueError):
        idx = 0
    return palette[idx]


def make_vertical_gradient(height: int, width: int, top: tuple[int, int, int], bottom: tuple[int, int, int]) -> np.ndarray:
    if height <= 0 or width <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    denom = max(1, height - 1)
    for y in range(height):
        alpha = y / denom
        color = (
            int(top[0] * (1 - alpha) + bottom[0] * alpha),
            int(top[1] * (1 - alpha) + bottom[1] * alpha),
            int(top[2] * (1 - alpha) + bottom[2] * alpha),
        )
        gradient[y, :] = color
    return gradient


def draw_glass_rect(img: np.ndarray, x: int, y: int, w: int, h: int, color: tuple[int, int, int], alpha: float) -> None:
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)


def put_text(img: np.ndarray, text: str, org: tuple[int, int], scale: float, color: tuple[int, int, int], thickness: int = 1) -> None:
    cv2.putText(img, text, org, FONT, scale, color, thickness, cv2.LINE_AA)


def format_compact(value: float, unit: str, decimals: int = 1) -> str:
    if not np.isfinite(value):
        value = 0.0
    abs_v = abs(value)
    suffix = ''
    if abs_v >= 1000:
        value = value / 1000.0
        suffix = 'k'
    fmt = f'{value:.{decimals}f}'
    if decimals > 0:
        fmt = fmt.rstrip('0').rstrip('.')
    value_text = f'{fmt}{suffix}'
    return f'{value_text} {unit}'.strip()


def fit_text_scale(text: str, max_width: int, base_scale: float, thickness: int) -> float:
    if max_width <= 0:
        return 0.3
    scale = base_scale
    (tw, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
    if tw <= max_width:
        return scale
    return max(0.3, scale * (max_width / max(tw, 1)))


def fit_text_scale_box(text: str, max_width: int, max_height: int, base_scale: float, thickness: int) -> float:
    if max_width <= 0 or max_height <= 0:
        return 0.3
    scale = fit_text_scale(text, max_width, base_scale, thickness)
    (_, th), _ = cv2.getTextSize(text, FONT, scale, thickness)
    if th <= max_height:
        return scale
    return max(0.3, scale * (max_height / max(th, 1)))


def split_value_unit(value: str) -> tuple[str, str]:
    parts = value.strip().split(' ')
    if len(parts) >= 2:
        return parts[0], ' '.join(parts[1:])
    return value, ''


def draw_card(img: np.ndarray, x: int, y: int, w: int, h: int, title: str, value: str, accent: tuple[int, int, int]) -> None:
    cv2.rectangle(img, (x, y), (x + w, y + h), (26, 30, 40), -1)
    cv2.rectangle(img, (x, y), (x + w, y + h), (50, 60, 80), 1)
    cv2.rectangle(img, (x, y), (x + w, y + 3), accent, -1)
    pad = 10
    title_region_h = int(h * 0.42)
    value_region_h = max(10, h - title_region_h - 8)
    title_scale = fit_text_scale_box(title, w - pad * 2, max(8, title_region_h - 4), 0.55, 1)
    value_text, unit_text = split_value_unit(value)
    unit_scale = 0.45 if unit_text else 0.0
    unit_w = 0
    if unit_text:
        (unit_w, _), _ = cv2.getTextSize(unit_text, FONT, unit_scale, 1)
    value_max_w = w - pad * 2 - (unit_w + 6 if unit_text else 0)
    value_scale = fit_text_scale_box(value_text, value_max_w, max(10, value_region_h - 4), 0.95, 2)
    (tw, th), _ = cv2.getTextSize(title, FONT, title_scale, 1)
    title_top = y + 6
    title_y = title_top + int((title_region_h + th) / 2)
    (vw, vh), _ = cv2.getTextSize(value_text, FONT, value_scale, 2)
    value_top = y + title_region_h + 6
    value_y = value_top + int((value_region_h + vh) / 2)
    cv2.line(img, (x + 6, y + title_region_h + 3), (x + w - 6, y + title_region_h + 3), (40, 48, 60), 1)
    put_text(img, title, (x + pad, title_y), title_scale, (170, 180, 200), 1)
    put_text(img, value_text, (x + pad, value_y), value_scale, (240, 240, 240), 2)
    if unit_text:
        put_text(img, unit_text, (x + w - pad - unit_w, value_y), unit_scale, (200, 210, 230), 1)


def draw_sparkline(img: np.ndarray, x: int, y: int, w: int, h: int, values: list[float], color: tuple[int, int, int]) -> None:
    if w <= 2 or h <= 2:
        return
    if len(values) < 2:
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 48, 60), 1)
        return
    vmin = min(values)
    vmax = max(values)
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax - vmin < 1e-6:
        vmin = 0.0
        vmax = 1.0
    pts = []
    n = len(values) - 1
    for i, v in enumerate(values):
        px = x + int((i / n) * (w - 1)) if n > 0 else x
        norm = (v - vmin) / (vmax - vmin)
        py = y + h - 2 - int(norm * (h - 4))
        pts.append((px, py))
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 48, 60), 1)
    for i in range(1, len(pts)):
        cv2.line(img, pts[i - 1], pts[i], color, 2)


def parse_roi(roi_text: str) -> tuple[float, float, float, float]:
    try:
        parts = [float(p.strip()) for p in roi_text.split(',')]
        if len(parts) != 4:
            raise ValueError
        x, y, w, h = parts
        if w <= 0 or h <= 0:
            raise ValueError
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        w = max(0.0, min(1.0 - x, w))
        h = max(0.0, min(1.0 - y, h))
        return x, y, w, h
    except Exception:
        return 0.08, 0.12, 0.84, 0.82


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Padel-lite tracking system')
    parser.add_argument('--input', required=True, help='Path to input video')
    parser.add_argument('--output', default='output/annotated.mp4', help='Path to output annotated video')
    parser.add_argument('--stats', default='output/stats.json', help='Path to output stats JSON')
    parser.add_argument('--model', default='yolov8n.pt', help='YOLO model path or name')
    parser.add_argument('--imgsz', type=int, default=640, help='YOLO inference size')
    parser.add_argument('--conf', type=float, default=0.25, help='YOLO confidence threshold')
    parser.add_argument('--iou', type=float, default=0.45, help='YOLO IoU threshold')
    parser.add_argument('--device', default='', help='YOLO device, e.g. 0 or cpu')
    parser.add_argument('--court-width-m', type=float, default=None, help='Court width in meters for scale')
    parser.add_argument('--court-width-px', type=float, default=None, help='Court width in pixels for scale')
    parser.add_argument('--draw-trajectories', action='store_true', help='Draw short trajectories')
    parser.add_argument('--dashboard', action='store_true', help='Render a stats panel below the video')
    parser.add_argument('--panel-height', type=int, default=340, help='Dashboard panel height in pixels')
    parser.add_argument('--max-tracks', type=int, default=4, help='Max tracks to show in the dashboard')
    parser.add_argument('--speed-ema', type=float, default=0.4, help='EMA factor for speed smoothing')
    parser.add_argument('--live', action='store_true', help='Show a live preview window while processing')
    parser.add_argument('--ignore-ids', default='', help='Comma-separated track IDs to ignore (e.g. 5,7)')
    parser.add_argument('--auto-filter', dest='auto_filter', action='store_true', help='Auto-filter non-player tracks')
    parser.add_argument('--no-auto-filter', dest='auto_filter', action='store_false', help='Disable auto-filter')
    parser.add_argument('--roi', default='0.08,0.12,0.84,0.82', help='Normalized ROI x,y,w,h for player area')
    parser.add_argument('--min-track-frames', type=int, default=12, help='Min frames before a track is shown')
    parser.add_argument('--min-roi-ratio', type=float, default=0.6, help='Min ROI ratio to keep a track')
    parser.add_argument('--keep-top', type=int, default=4, help='Keep top N tracks by distance among candidates')
    parser.set_defaults(auto_filter=True)
    return parser.parse_args()


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    stats_path = Path(args.stats)
    ensure_parent(output_path)
    ensure_parent(stats_path)

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f'Failed to open video: {input_path}')
        return 1

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    panel_h = args.panel_height if args.dashboard else 0
    out_size = (width, height + panel_h)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, out_size)

    model = YOLO(args.model)

    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_iou_distance=0.7,
        max_cosine_distance=0.4,
        nn_budget=None,
    )

    meter_per_px = None
    if args.court_width_m is not None and args.court_width_px is not None:
        if args.court_width_px > 0:
            meter_per_px = args.court_width_m / args.court_width_px

    roi_norm = parse_roi(args.roi)
    roi_px = (
        int(roi_norm[0] * width),
        int(roi_norm[1] * height),
        int(roi_norm[2] * width),
        int(roi_norm[3] * height),
    )

    ignore_ids = set()
    if args.ignore_ids:
        for token in args.ignore_ids.split(','):
            token = token.strip()
            if not token:
                continue
            try:
                ignore_ids.add(int(token))
            except ValueError:
                continue

    last_center = {}
    trajectories = {}
    total_px = {}
    track_frames = {}
    track_speed = {}
    track_max_speed = {}
    track_moving_frames = {}
    track_sprint_count = {}
    track_in_sprint = {}
    track_max_accel = {}
    track_prev_speed = {}
    track_lr = {}
    track_fb = {}
    track_roi_frames = {}
    speed_history = deque(maxlen=240)
    heatmap = None
    heatmap_w = 0
    heatmap_h = 0
    frame_idx = 0

    window_name = 'Padel-lite' if args.live else None
    if args.live:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)

    panel_bg = None
    if args.dashboard:
        panel_bg = make_vertical_gradient(panel_h, width, (18, 22, 32), (10, 12, 18))
        for x in range(0, width, 120):
            cv2.line(panel_bg, (x, 0), (x, panel_h), (22, 26, 36), 1)
        for y in range(0, panel_h, 60):
            cv2.line(panel_bg, (0, y), (width, y), (22, 26, 36), 1)
        heatmap_w = 48
        heatmap_h = 27
        heatmap = np.zeros((heatmap_h, heatmap_w), dtype=np.float32)
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if heatmap is not None:
            heatmap *= 0.995

        frame_idx += 1
        results = model(
            frame,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            device=args.device,
            verbose=False,
        )[0]

        detections = []
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                cls_id = int(box.cls[0])
                if cls_id != 0:
                    continue
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], conf, 'person'))

        tracks = tracker.update_tracks(detections, frame=frame)
        draw_items = []
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                pass
            if track_id in ignore_ids:
                continue
            x1, y1, x2, y2 = track.to_ltrb()
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0

            dist = 0.0
            if track_id in last_center:
                dx = cx - last_center[track_id][0]
                dy = cy - last_center[track_id][1]
                dist = float(np.hypot(dx, dy))
                total_px[track_id] = total_px.get(track_id, 0.0) + dist
            else:
                total_px.setdefault(track_id, 0.0)
            last_center[track_id] = (cx, cy)
            track_frames[track_id] = track_frames.get(track_id, 0) + 1

            speed_px_s = dist * fps
            prev_speed = track_speed.get(track_id, speed_px_s)
            smooth_speed = (1.0 - args.speed_ema) * prev_speed + args.speed_ema * speed_px_s
            track_speed[track_id] = smooth_speed
            track_max_speed[track_id] = max(track_max_speed.get(track_id, 0.0), smooth_speed)

            move_thresh = 5.0
            if smooth_speed > move_thresh:
                track_moving_frames[track_id] = track_moving_frames.get(track_id, 0) + 1
            prev = track_prev_speed.get(track_id, smooth_speed)
            accel = (smooth_speed - prev) * fps
            track_prev_speed[track_id] = smooth_speed
            track_max_accel[track_id] = max(track_max_accel.get(track_id, 0.0), abs(accel))
            sprint_thresh = max(0.7 * track_max_speed.get(track_id, 0.0), move_thresh * 3.0)
            in_sprint = smooth_speed >= sprint_thresh and smooth_speed > 0
            if in_sprint and not track_in_sprint.get(track_id, False):
                track_sprint_count[track_id] = track_sprint_count.get(track_id, 0) + 1
            track_in_sprint[track_id] = in_sprint

            lr = track_lr.setdefault(track_id, [0, 0])
            if cx < width * 0.5:
                lr[0] += 1
            else:
                lr[1] += 1
            fb = track_fb.setdefault(track_id, [0, 0])
            if cy < height * 0.5:
                fb[0] += 1
            else:
                fb[1] += 1

            if args.auto_filter:
                rx, ry, rw, rh = roi_px
                if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
                    track_roi_frames[track_id] = track_roi_frames.get(track_id, 0) + 1

            segment = None
            if args.draw_trajectories:
                path = trajectories.setdefault(track_id, [])
                path.append((cx, cy))
                if len(path) >= 2:
                    p1 = (int(path[-2][0]), int(path[-2][1]))
                    p2 = (int(path[-1][0]), int(path[-1][1]))
                    segment = (p1, p2)

            draw_items.append((track_id, x1, y1, x2, y2, cx, cy, segment))

        visible_ids = set()
        if args.auto_filter:
            candidates = []
            for tid, frames in track_frames.items():
                if tid in ignore_ids:
                    continue
                if frames < args.min_track_frames:
                    continue
                ratio = track_roi_frames.get(tid, 0) / max(1, frames)
                if ratio < args.min_roi_ratio:
                    continue
                candidates.append(tid)
            if args.keep_top > 0 and len(candidates) > args.keep_top:
                candidates = sorted(candidates, key=lambda t: total_px.get(t, 0.0), reverse=True)[: args.keep_top]
            visible_ids = set(candidates)
        else:
            visible_ids = {tid for tid in track_frames.keys() if tid not in ignore_ids}

        for track_id, x1, y1, x2, y2, cx, cy, segment in draw_items:
            if track_id not in visible_ids:
                continue
            if args.draw_trajectories and segment is not None:
                cv2.line(frame, segment[0], segment[1], color_for_id(track_id), 2)
            color = color_for_id(track_id)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f'ID {track_id}'
            (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 2)
            lx, ly = int(x1), max(0, int(y1) - 22)
            draw_glass_rect(frame, lx, ly, tw + 10, 20, color, 0.55)
            cv2.putText(frame, label, (lx + 5, ly + 15), FONT, 0.6, (20, 20, 20), 2)

            if heatmap is not None and width > 0 and height > 0:
                ix = int(np.clip(cx / width * heatmap_w, 0, heatmap_w - 1))
                iy = int(np.clip(cy / height * heatmap_h, 0, heatmap_h - 1))
                heatmap[iy, ix] += 1.0

        overlay_text = f'Frame {frame_idx}'
        cv2.putText(frame, overlay_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        if args.dashboard:
            canvas = np.zeros((height + panel_h, width, 3), dtype=np.uint8)
            canvas[:height] = frame
            panel = canvas[height:]
            panel[:] = panel_bg if panel_bg is not None else (18, 20, 26)
            header_h = 40
            pad_x = 16
            time_sec = frame_idx / fps

            draw_glass_rect(panel, 0, 0, width, header_h, (26, 30, 42), 0.8)
            put_text(panel, 'PADEL-LITE | MATCH INSIGHTS', (pad_x, 26), 0.65, (235, 240, 255), 2)
            unit_label = 'm' if meter_per_px is not None else 'px'
            header_text = f't={time_sec:0.1f}s  fps={fps:0.1f}  unit={unit_label}'
            header_scale = fit_text_scale(header_text, max(120, width - pad_x * 2 - 260), 0.5, 1)
            (htw, _), _ = cv2.getTextSize(header_text, FONT, header_scale, 1)
            put_text(panel, header_text, (width - pad_x - htw, 26), header_scale, (200, 200, 200), 1)

            rows = [(tid, total_px.get(tid, 0.0)) for tid in total_px.keys() if tid in visible_ids]
            rows = sorted(rows, key=lambda kv: kv[1], reverse=True)
            active_tracks = len(rows)
            total_dist_px = sum(dist for _, dist in rows)
            if meter_per_px is not None:
                total_dist_label = format_compact(total_dist_px * meter_per_px, 'm', 1)
            else:
                total_dist_label = format_compact(total_dist_px, 'px', 0)

            max_speed_ref = max((track_max_speed.get(tid, 0.0) for tid, _ in rows), default=0.0)
            if not np.isfinite(max_speed_ref) or max_speed_ref <= 1e-6:
                max_speed_ref = 1.0
            max_speed_label = format_compact(max_speed_ref * meter_per_px, 'm/s', 2) if meter_per_px is not None else format_compact(max_speed_ref, 'px/s', 1)
            if rows:
                avg_speed_all = sum(track_speed.get(tid, 0.0) for tid, _ in rows) / max(1, len(rows))
            else:
                avg_speed_all = 0.0
            avg_speed_label = format_compact(avg_speed_all * meter_per_px, 'm/s', 2) if meter_per_px is not None else format_compact(avg_speed_all, 'px/s', 1)
            speed_history.append(avg_speed_all)

            moving_ratios = []
            for tid, _ in rows:
                frames = track_frames.get(tid, 0)
                if frames <= 0:
                    continue
                moving_ratios.append(track_moving_frames.get(tid, 0) / frames)
            moving_pct = (sum(moving_ratios) / max(1, len(moving_ratios))) * 100.0 if moving_ratios else 0.0
            sprint_total = sum(track_sprint_count.get(tid, 0) for tid, _ in rows) if rows else 0
            top_accel = max((track_max_accel.get(tid, 0.0) for tid, _ in rows), default=0.0)
            if meter_per_px is not None:
                top_accel_label = format_compact(top_accel * meter_per_px, 'm/s^2', 2)
            else:
                top_accel_label = format_compact(top_accel, 'px/s^2', 1)
            left_total = sum(track_lr.get(tid, [0, 0])[0] for tid, _ in rows) if rows else 0
            right_total = sum(track_lr.get(tid, [0, 0])[1] for tid, _ in rows) if rows else 0
            lr_total = left_total + right_total
            if lr_total > 0:
                left_pct = 100.0 * left_total / lr_total
                right_pct = 100.0 - left_pct
                lr_label = f'{left_pct:0.0f}/{right_pct:0.0f}%'
            else:
                lr_label = '0/0%'

            card_y = header_h + 10
            card_h = 52
            card_gap = 10
            card_w = int((width - pad_x * 2 - card_gap * 3) / 4)
            if card_w < 120:
                card_gap = 6
                card_w = int((width - pad_x * 2 - card_gap * 3) / 4)
            card_w = max(60, card_w)
            draw_card(panel, pad_x, card_y, card_w, card_h, 'ACTIVE', str(active_tracks), (80, 180, 255))
            draw_card(panel, pad_x + card_w + card_gap, card_y, card_w, card_h, 'TOTAL DIST', total_dist_label, (80, 255, 170))
            draw_card(panel, pad_x + 2 * (card_w + card_gap), card_y, card_w, card_h, 'AVG SPEED', avg_speed_label, (255, 160, 80))
            draw_card(panel, pad_x + 3 * (card_w + card_gap), card_y, card_w, card_h, 'TOP SPEED', max_speed_label, (190, 120, 255))

            sub_y = card_y + card_h + 8
            sub_h = 40
            sub_gap = 10
            sub_w = int((width - pad_x * 2 - sub_gap * 3) / 4)
            sub_w = max(60, sub_w)
            draw_card(panel, pad_x, sub_y, sub_w, sub_h, 'MOTION %', f'{moving_pct:0.0f}%', (90, 200, 255))
            draw_card(panel, pad_x + sub_w + sub_gap, sub_y, sub_w, sub_h, 'SPRINTS', str(sprint_total), (255, 210, 90))
            draw_card(panel, pad_x + 2 * (sub_w + sub_gap), sub_y, sub_w, sub_h, 'TOP ACCEL', top_accel_label, (255, 120, 140))
            draw_card(panel, pad_x + 3 * (sub_w + sub_gap), sub_y, sub_w, sub_h, 'L/R BAL', lr_label, (130, 160, 255))

            chart_y = sub_y + sub_h + 18
            chart_h = 64
            chart_x = pad_x
            chart_w = int(width * 0.62) - pad_x
            chart_w = max(120, chart_w)
            heat_x = chart_x + chart_w + 12
            heat_w = width - pad_x - heat_x
            heat_w = max(80, heat_w)

            put_text(panel, 'MOTION TREND', (chart_x, chart_y - 4), 0.45, (140, 160, 190), 1)
            draw_sparkline(panel, chart_x, chart_y, chart_w, chart_h, list(speed_history), (80, 180, 255))

            if heatmap is not None and heat_w > 10 and chart_h > 10:
                hm = heatmap.copy()
                if hm.max() > 0:
                    hm = hm / hm.max()
                hm_img = (hm * 255).astype(np.uint8)
                hm_color = cv2.applyColorMap(hm_img, cv2.COLORMAP_TURBO)
                hm_resized = cv2.resize(hm_color, (heat_w, chart_h), interpolation=cv2.INTER_LINEAR)
                panel[chart_y:chart_y + chart_h, heat_x:heat_x + heat_w] = hm_resized
                cv2.rectangle(panel, (heat_x, chart_y), (heat_x + heat_w, chart_y + chart_h), (40, 48, 60), 1)
                put_text(panel, 'HEATMAP', (heat_x, chart_y - 4), 0.45, (140, 160, 190), 1)

            table_top = chart_y + chart_h + 16
            row_h = max(24, int((panel_h - table_top - 8) / max(1, args.max_tracks)))
            col_id = pad_x
            col_dist = int(width * 0.16)
            col_avg = int(width * 0.34)
            col_cur = int(width * 0.48)
            col_bar = int(width * 0.62)
            col_max = int(width * 0.85)
            bar_w = max(80, int(width * 0.18))

            put_text(panel, 'ID', (col_id, table_top - 6), 0.5, (140, 160, 190), 1)
            put_text(panel, 'Distance', (col_dist, table_top - 6), 0.5, (140, 160, 190), 1)
            put_text(panel, 'Avg', (col_avg, table_top - 6), 0.5, (140, 160, 190), 1)
            put_text(panel, 'Cur', (col_cur, table_top - 6), 0.5, (140, 160, 190), 1)
            put_text(panel, 'Speed', (col_bar, table_top - 6), 0.5, (140, 160, 190), 1)
            put_text(panel, 'Max', (col_max, table_top - 6), 0.5, (140, 160, 190), 1)

            y = table_top
            for idx, (track_id, dist_px) in enumerate(rows[: args.max_tracks]):
                if idx % 2 == 0:
                    cv2.rectangle(panel, (0, y - row_h + 4), (width, y + 4), (22, 24, 32), -1)
                frames = track_frames.get(track_id, 1)
                avg_speed = (dist_px / max(frames, 1)) * fps
                cur_speed = track_speed.get(track_id, 0.0)
                max_speed = track_max_speed.get(track_id, 0.0)
                if meter_per_px is not None:
                    dist_label = f'{dist_px * meter_per_px:0.1f} m'
                    avg_label = f'{avg_speed * meter_per_px:0.2f}'
                    cur_label = f'{cur_speed * meter_per_px:0.2f}'
                    max_label = f'{max_speed * meter_per_px:0.2f}'
                else:
                    dist_label = f'{dist_px:0.0f} px'
                    avg_label = f'{avg_speed:0.1f}'
                    cur_label = f'{cur_speed:0.1f}'
                    max_label = f'{max_speed:0.1f}'

                row_color = color_for_id(track_id)
                put_text(panel, str(track_id), (col_id, y), 0.55, row_color, 2)
                put_text(panel, dist_label, (col_dist, y), 0.55, (220, 220, 220), 1)
                put_text(panel, avg_label, (col_avg, y), 0.55, (220, 220, 220), 1)
                put_text(panel, cur_label, (col_cur, y), 0.55, (220, 220, 220), 1)
                denom = max_speed_ref if max_speed_ref > 0 else 1.0
                ratio = cur_speed / denom
                if not np.isfinite(ratio):
                    ratio = 0.0
                ratio = min(max(ratio, 0.0), 1.0)
                bar_fill = int(bar_w * ratio)
                cv2.rectangle(panel, (col_bar, y - 10), (col_bar + bar_w, y + 4), (40, 48, 60), -1)
                cv2.rectangle(panel, (col_bar, y - 10), (col_bar + bar_fill, y + 4), row_color, -1)
                put_text(panel, max_label, (col_max, y), 0.55, (220, 220, 220), 1)
                y += row_h

            rendered = canvas
        else:
            rendered = frame

        writer.write(rendered)
        if args.live:
            cv2.imshow(window_name, rendered)
            delay = max(1, int(1000 / max(fps, 1.0)))
            key = cv2.waitKey(delay) & 0xFF
            if key in (ord('q'), 27):
                break

    cap.release()
    writer.release()
    if args.live:
        cv2.destroyAllWindows()

    if args.auto_filter:
        candidates = []
        for tid, frames in track_frames.items():
            if tid in ignore_ids:
                continue
            if frames < args.min_track_frames:
                continue
            ratio = track_roi_frames.get(tid, 0) / max(1, frames)
            if ratio < args.min_roi_ratio:
                continue
            candidates.append(tid)
        if not candidates:
            candidates = list(track_frames.keys())
        if args.keep_top > 0 and len(candidates) > args.keep_top:
            candidates = sorted(candidates, key=lambda t: total_px.get(t, 0.0), reverse=True)[: args.keep_top]
        final_visible = set(candidates)
    else:
        final_visible = {tid for tid in track_frames.keys() if tid not in ignore_ids}

    stats = {
        'input': str(input_path),
        'output': str(output_path),
        'fps': fps,
        'total_frames': frame_idx,
        'tracks': [],
    }

    for track_id, dist_px in sorted(total_px.items()):
        if track_id not in final_visible:
            continue
        frames = track_frames.get(track_id, 1)
        avg_speed = (dist_px / max(frames, 1)) * fps
        moving_pct = track_moving_frames.get(track_id, 0) / max(frames, 1)
        entry = {
            'track_id': int(track_id) if isinstance(track_id, (int, np.integer)) else str(track_id),
            'distance_px': dist_px,
            'avg_speed_px_s': avg_speed,
            'max_speed_px_s': track_max_speed.get(track_id, 0.0),
            'moving_pct': moving_pct,
            'sprint_count': track_sprint_count.get(track_id, 0),
            'max_accel_px_s2': track_max_accel.get(track_id, 0.0),
        }
        if meter_per_px is not None:
            entry['distance_m'] = dist_px * meter_per_px
            entry['avg_speed_m_s'] = avg_speed * meter_per_px
            entry['max_speed_m_s'] = track_max_speed.get(track_id, 0.0) * meter_per_px
            entry['max_accel_m_s2'] = track_max_accel.get(track_id, 0.0) * meter_per_px
        lr = track_lr.get(track_id, [0, 0])
        lr_total = lr[0] + lr[1]
        if lr_total > 0:
            entry['left_pct'] = lr[0] / lr_total
            entry['right_pct'] = lr[1] / lr_total
        stats['tracks'].append(entry)

    stats_path.write_text(json.dumps(stats, indent=2))

    print(f'Wrote annotated video to: {output_path}')
    print(f'Wrote stats to: {stats_path}')
    if meter_per_px is None:
        print('Distances are in pixels. Provide --court-width-m and --court-width-px to convert to meters.')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
