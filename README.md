# Bullseye Marker Motion Tracking using OpenCV

A computer vision pipeline for accurately tracking the vertical motion of a circular bullseye marker from video.

The program automatically detects the marker, tracks it throughout the video using template matching and a Kalman filter, converts pixel displacement into physical displacement (mm), estimates vertical velocity, and exports all measurements to CSV for comparison with theoretical mechanism motion.

---

# Features

- Automatic bullseye marker detection
- Concentric circle recognition
- Automatic pixel-to-millimeter calibration
- Local template tracking
- Kalman filter prediction
- Automatic recovery if tracking is lost
- Physical displacement measurement
- Vertical velocity estimation
- CSV export
- Annotated output video
- Optional comparison with theoretical slider displacement

---

# Marker Specification

The tracker expects a marker similar to "marker_template.png"

Specifically,

- White square background
- Black filled center circle
- Two concentric circular rings

The outermost ring is used for calibration.

---

# Folder Structure

```
Project/
│
├── track.py
├── marker_template.png
├── input_video.mov
│
├── tracked_bullseye_fixed.mp4
├── tracked_height_data.csv
├── vertical_velocity.csv
│
└── README.md
```

---

# Requirements

Python 3.9+

Install dependencies

```bash
pip install opencv-python numpy
```

---

# User Configuration

All user-editable parameters are located at the top of the script.

---

## Input Video

```python
VIDEO_PATH
```

Path to the input video.

Example

```python
VIDEO_PATH = r"C:\Videos\experiment.mov"
```

---

## Template

```python
TEMPLATE_PATH
```

Reference image of the bullseye marker.

Used only if automatic structural detection fails.

---

## Output Files

```python
OUTPUT_VIDEO
```

Annotated tracking video.

Example

```
tracked_bullseye_fixed.mp4
```

---

```python
OUTPUT_CSV
```

Contains

- Frame number
- Time
- Marker position
- Height
- Detection confidence

---

```python
VELOCITY_CSV
```

Contains

- Height
- Vertical velocity
- Reference slider displacement

---

# Calibration

The code converts pixels into millimeters.

Two calibration methods exist.

---

## Primary Calibration

```python
OUTER_CIRCLE_DIAMETER_MM
```

Physical diameter of the outermost ring.

Example

```python
OUTER_CIRCLE_DIAMETER_MM = 11
```

Scale is computed as

```
mm_per_pixel

=

Outer Diameter (mm)
-------------------------
Detected Diameter (pixels)
```

---

## Fallback Calibration

If the circle detector fails,

```python
REFERENCE_TEMPLATE_WIDTH_MM
```

is used.

This should be the physical width of the template image.

---

# Motion Direction

```python
HEIGHT_POSITIVE_DIR
```

Options

```
"down"
```

Positive displacement occurs when the marker moves downward.

or

```
"up"
```

Positive displacement occurs upward.

Choose according to your experimental setup.

---

# Theoretical Model

The tracker can optionally compare measured displacement with the slider-crank equation.

Enable

```python
EXPORT_REFERENCE_MODEL = True
```

Model parameters

```python
MODEL_C0_MM

MODEL_R_MM

MODEL_L_MM
```

Correspond to

- Initial slider position
- Crank radius
- Connecting rod length

---

# Reference Crank Angle

The program assumes the video corresponds to

```python
THETA_START_DEG
```

↓

```python
THETA_END_DEG
```

Example

```python
38°

↓

90°
```

Each frame is assigned an interpolated crank angle.

---

# Structural Detection Parameters

The first frame is analyzed to locate concentric circles.

---

## MIN_CIRCULARITY

Rejects shapes that are insufficiently circular.

Higher values

- stricter detection

Lower values

- accepts noisier contours

Default

```python
0.55
```

---

## MIN_RADIUS_PX

Minimum acceptable circle radius.

Small noise blobs are ignored.

---

## MAX_RADIUS_FRAC

Maximum circle radius as a fraction of image size.

Prevents selecting very large contours.

---

## CONCENTRIC_TOL

Maximum allowed distance between centers of concentric circles.

Default

```python
14 pixels
```

---

## MIN_RINGS

Minimum concentric circles required.

Default

```
2
```

---

# Template Matching

After initialization, tracking uses template matching.

---

## MATCH_THRESH

Minimum correlation score.

Higher

- fewer false detections

Lower

- more tolerant

Default

```
0.30
```

---

## SEARCH_MARGIN

Search window around predicted position.

Larger values

- slower
- more robust

Smaller values

- faster

---

## ROI_MATCH_SCALES

Template scales tested.

Example

```python
[0.85,1.0,1.15]
```

Allows small scale changes.

---

# Re-detection

```python
LOST_THRESH
```

If tracking fails for this many frames,

the program attempts full detection again.

---

# Kalman Filter

State vector

```
x

y

vx

vy
```

Prediction model

Constant velocity.

Parameters

```python
KF_PROC_NOISE
```

Controls responsiveness.

Higher

- reacts faster
- noisier

Lower

- smoother

---

```python
KF_MEAS_NOISE
```

Measurement uncertainty.

Increase if detections are noisy.

---

# Processing Pipeline

```
Input Video

        │

        ▼

Initial Structural Detection

        │

        ▼

Calibration

        │

        ▼

Extract Local Template

        │

        ▼

Kalman Prediction

        │

        ▼

ROI Template Matching

        │

        ▼

Kalman Update

        │

        ▼

Height Calculation

        │

        ▼

Velocity Estimation

        │

        ▼

CSV Export

        │

        ▼

Annotated Video
```

---

# Output Files

## tracked_bullseye_fixed.mp4

Annotated tracking video showing

- Search window
- Marker position
- Detection confidence
- Height
- Reference angle
- Calibration scale

---

## tracked_height_data.csv

Columns

```
frame

time_s

theta_ref_deg

cx_px

cy_px

height_mm

confidence

detected

slider_ref_rel_mm
```

---

## vertical_velocity.csv

Columns

```
time_s

theta_ref_deg

height_mm

vertical_velocity_m_per_s

detected

slider_ref_rel_mm
```

---

# How Height is Computed

The initial marker position is treated as zero.

Each frame

```
Height

=

(Current Y

−

Initial Y)

×

mm_per_pixel
```

or the opposite depending on

```
HEIGHT_POSITIVE_DIR
```

---

# Velocity Estimation

Velocity is computed numerically using

```
numpy.gradient()
```

and converted to

```
m/s
```

---

# Limitations

This tracker assumes

- Single visible marker
- Marker remains approximately front-facing
- Small scale variation
- Limited rotation
- Stable illumination
- Marker stays inside the frame
- Camera remains fixed

---

# Future Improvements

Possible future enhancements include

- Subpixel circle fitting
- ArUco integration
- Optical flow tracking
- Perspective correction
- Adaptive template updating
- Savitzky-Golay velocity smoothing
- Camera calibration
- Multi-marker tracking
- GPU acceleration
- Real-time processing

---

# License

This project is intended for academic and research purposes.
