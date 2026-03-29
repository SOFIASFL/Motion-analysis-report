# Padel-Lite Tracking System

End-to-end demo of a **padel player analytics pipeline**: YOLO detection → DeepSORT tracking → live dashboard → annotated video + stats JSON. Built to be **demo-ready in under an hour**, but structured so it can evolve into production.

## Why This Is Interesting (Recruiter-Friendly)
- Real-time CV pipeline with detection + multi-object tracking.
- Persistent IDs and trajectories across frames.
- Derived analytics (distance, speed, acceleration, motion %, sprint count).
- Live dashboard with KPI cards, motion trend, and heatmap.
- Production mindset: clear extension path for calibration, ball tracking, and scoring events.

## Quick Start
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python padel_lite.py --input path\to\video.mp4 --output output\annotated.mp4 --stats output\stats.json --dashboard --live
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

## Features
- Detection: YOLOv8 (fast baseline)
- Tracking: DeepSORT (stable IDs)
- Dashboard:
  - KPI cards (active, total distance, avg speed, top speed)
  - Secondary KPIs (motion %, sprints, top accel, left/right balance)
  - Motion trend sparkline
  - Court heatmap
- Stats export: `output/stats.json` per-track metrics

## Usage Examples
Basic run:
```powershell
python padel_lite.py --input path\to\video.mp4 --dashboard
```

Live preview (quit with `q` or `Esc`):
```powershell
python padel_lite.py --input path\to\video.mp4 --dashboard --live
```

Ignore a known non-player track:
```powershell
python padel_lite.py --input path\to\video.mp4 --ignore-ids 5
```

Auto-filter non-player tracks (default on), tune ROI/thresholds:
```powershell
python padel_lite.py --input path\to\video.mp4 --auto-filter --roi 0.08,0.12,0.84,0.82 --min-track-frames 12 --min-roi-ratio 0.6 --keep-top 4
```

Disable auto-filter:
```powershell
python padel_lite.py --input path\to\video.mp4 --no-auto-filter
```

## Distance Scale (Optional)
Convert pixel distance to meters using court width:
```powershell
python padel_lite.py --input path\to\video.mp4 --court-width-m 10.0 --court-width-px 800
```

## Output
- `output/annotated.mp4` — video with boxes, IDs, and dashboard
- `output/stats.json` — per-track analytics (distance, speed, accel, motion %, sprints, L/R balance)

## Road To Production
**Phase 1 — Reliability**
- Camera calibration + homography for real-world meters
- Track re-identification across long occlusions
- Court ROI auto-detection to filter non-players

**Phase 2 — Deeper Analytics**
- Ball tracking + shot classification (serve, volley, smash)
- Rally segmentation + winner/error attribution
- Player-specific profiles and match summaries

**Phase 3 — Deployment**
- GPU inference optimization (TensorRT / ONNX)
- Real-time pipeline (frame batching, async queues)
- Web dashboard + API for downstream analytics

## Notes
- Default model: `yolov8n.pt` (fast). Try `--model yolov8s.pt` for accuracy.
- First run may download model weights.
- Stats are per-track (IDs are not player-identity stable across matches).
