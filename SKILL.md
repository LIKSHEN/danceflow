---
name: danceflow
description: "Accept a still image of all dancers in their starting formation + a song (MP3, name, or link) and produce a full timestamped choreography movement plan for every dancer. Outputs an interactive HTML dashboard (default) or PDF (on request) delivered via Telegram. Uses person detection, pose estimation, spatial layout analysis, and song rhythm/structure analysis."
tags: [dance, choreography, pose-estimation, person-detection, rhythm-analysis, movement-planning, telegram, dashboard, music]
version: 1.0
---

# DanceFlow — Formation-to-Choreography Planner

## 1. Skill Name

**DanceFlow** — an AI agent skill that analyses a starting dancer formation image and a song to generate a full timestamped choreography movement plan for every detected dancer, delivered as an interactive HTML dashboard or PDF via Telegram.

---

## 2. Target User

| User Type | Use Case |
|-----------|----------|
| Dance instructors / choreographers | Plan and document full-song movement sequences before rehearsal |
| Student dance groups | Get an AI-assisted movement plan for competition or event performances |
| Performing arts students | Learn how to structure choreography relative to song structure |
| Event and production managers | Review dance acts and plan stage usage efficiently |
| Music and performing arts lecturers | Demonstrate how rhythm maps to movement in teaching |

**Primary target:** Non-technical dance practitioners who need structured choreography plans but do not have formal notation training or choreography software.

---

## 3. Real-World Problem

Planning choreography for a group of dancers is time-consuming and mentally demanding. A choreographer must simultaneously:

1. Track multiple dancers across the whole stage at all times.
2. Match movement cues to the song's tempo, rhythm, and section changes (verse, chorus, bridge, drop).
3. Ensure formations change smoothly without collision or visual imbalance.
4. Communicate the plan clearly to every dancer by name or position.

Without a structured plan, rehearsals are inefficient. Dancers repeat the same sections, movements are verbally described and easily forgotten, and formation transitions are improvised rather than designed.

**DanceFlow** solves this by:
- Detecting all dancers and their starting positions from a single photo.
- Analysing the song structure and tempo from the uploaded MP3 or song name.
- Generating a complete, timestamped movement plan for every dancer.
- Presenting the plan as a visual HTML dashboard or PDF that any dancer can read without technical knowledge.

---

## 4. Input Format

### Primary Input A — Dancer Formation Image

| Property | Details |
|----------|---------|
| Format | JPG, PNG, WebP |
| Content | A still photo of all dancers standing in their starting positions on stage or rehearsal floor |
| Quality requirement | Must show full or near-full body of all dancers; overhead or frontal view preferred |
| Minimum resolution | 640 × 480 px recommended |
| Accepted views | Frontal stage view, slight elevated angle, top-down floor view |

**What the agent extracts from the image:**
- Total number of dancers detected.
- Relative X/Y position of each dancer on the stage plane (normalised 0–1 grid).
- Rough body orientation (facing front, side, or back).
- Initial formation type (line, V-shape, cluster, grid, diagonal, scattered).
- Image quality indicators: blur score, brightness, occlusion level.

### Primary Input B — Song

| Method | Format | Notes |
|--------|--------|-------|
| MP3 upload | `.mp3` file via Telegram | Agent analyses audio for BPM, key sections |
| Song name | Plain text, e.g. `"Shape of You – Ed Sheeran"` | Agent uses known song structure data |
| Song link | YouTube, Spotify, or SoundCloud URL | Agent fetches metadata and structure |

**What the agent extracts from the song:**
- BPM (beats per minute).
- Total song duration.
- Song sections with timestamps: intro, verse, pre-chorus, chorus, bridge, drop, outro.
- Rhythmic character per section (slow/fast, smooth/sharp, heavy/light).
- Suggested movement energy level per section (calm, moderate, high-energy, climax).

### Optional Metadata (user may provide in Telegram message)

| Field | Example | Effect |
|-------|---------|--------|
| Dance style hint | `"contemporary"`, `"hip-hop"`, `"K-pop"`, `"ballet"` | Adjusts movement vocabulary used in the plan |
| Dancer labels | `"Dancer 1 = Aisha, Dancer 2 = Ben"` | Personalises the output with names instead of numbered labels |
| Stage dimensions | `"8m wide x 6m deep"` | Scales the position grid to actual measurements |
| Output format | `"PDF"` | Produces PDF output instead of the default HTML dashboard |
| Number of sections | `"focus on chorus only"` | Restricts output to a specific part of the song |

---

## 5. CV / Image-Processing Method

### A. Image Preprocessing Pipeline

Raw photos taken in rehearsal spaces are often poorly lit, slightly skewed, or cluttered with background objects. The following preprocessing steps are applied before any detection task.

| Step | Method | Purpose |
|------|--------|---------|
| 1. Blur detection | Laplacian variance score | Flag image if variance < 100; warn user if too blurry |
| 2. Brightness normalisation | Histogram equalisation (CLAHE) | Correct underexposed or overexposed rehearsal photos |
| 3. Grayscale conversion | OpenCV `cvtColor(BGR2GRAY)` | Reduce noise before edge and contour detection |
| 4. Denoising | `fastNlMeansDenoising` | Remove compression artefacts and grain |
| 5. Contrast enhancement | Adaptive thresholding or CLAHE | Improve body-background separation |
| 6. Background segmentation | GrabCut or thresholding | Isolate the stage floor from walls and props |
| 7. Resolution upscale | Bicubic interpolation if < 640px height | Ensure person detector has sufficient pixel detail |

```python
import cv2
import numpy as np

def preprocess_formation_image(image_path):
    img = cv2.imread(image_path)

    # Step 1 — Blur detection
    gray_raw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray_raw, cv2.CV_64F).var()
    if blur_score < 100:
        print(f"WARNING: Image may be too blurry (score={blur_score:.1f}). Results may be less accurate.")

    # Step 2 — Brightness normalisation
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)

    # Step 3+4 — Grayscale + Denoising
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.fastNlMeansDenoising(gray, h=10)

    # Step 7 — Resolution upscale if needed
    h, w = img.shape[:2]
    if h < 640:
        scale = 640 / h
        img = cv2.resize(img, (int(w * scale), 640), interpolation=cv2.INTER_CUBIC)

    return img, blur_score
```

---

### B. Person Detection — Counting and Locating All Dancers

**Method:** YOLOv8 (pretrained COCO model) or MediaPipe Person Segmentation.

- Detect all `person` class bounding boxes in the image.
- Filter out detections with confidence < 0.5 to reduce false positives (chairs, props).
- Assign each dancer a numbered label: `D1`, `D2`, `D3`, etc., ordered left-to-right by bounding box centre X.
- Record normalised stage coordinates `(x_norm, y_norm)` where `(0,0)` = top-left and `(1,1)` = bottom-right of the detected stage floor region.

```python
# Pseudocode — YOLOv8 detection
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
results = model(preprocessed_image)

dancers = []
for box in results[0].boxes:
    if box.cls == 0 and box.conf > 0.5:   # class 0 = person
        cx = (box.xyxy[0] + box.xyxy[2]) / 2   # centre X
        cy = (box.xyxy[1] + box.xyxy[3]) / 2   # centre Y
        dancers.append({"id": None, "cx": cx, "cy": cy, "bbox": box.xyxy})

# Sort left-to-right, assign IDs
dancers.sort(key=lambda d: d["cx"])
for i, d in enumerate(dancers):
    d["id"] = f"D{i+1}"
```

---

### C. Pose Estimation — Initial Body Orientation

**Method:** MediaPipe Pose (single-frame, multi-person via bounding box crop).

For each detected dancer bounding box:
1. Crop the image to the bounding box region (with 10% padding).
2. Run MediaPipe Pose on the crop.
3. Extract key landmarks: shoulders, hips, knees, feet, wrists.
4. Determine facing direction: frontal, side-on, or back-facing (by comparing shoulder and hip keypoint visibility scores).
5. Detect initial posture: standing upright, crouched, arms raised, arms open, etc.

**Output per dancer:**

```json
{
  "id": "D1",
  "position": {"x_norm": 0.22, "y_norm": 0.60},
  "facing": "front",
  "posture": "standing_arms_down",
  "confidence": 0.87
}
```

---

### D. Formation Analysis — Spatial Layout Recognition

**Method:** Geometric clustering + pattern matching against known formation templates.

After all dancer positions are recorded, the agent maps their `(x_norm, y_norm)` coordinates and classifies the formation:

| Formation Name | Description | Detection Method |
|----------------|-------------|-----------------|
| Straight line | All dancers along one horizontal or vertical axis | Linear regression R² > 0.95 |
| V-shape / wedge | Lead dancer forward, two wings extending back | Triangle vertex detection |
| Grid / matrix | Rows and columns of equal spacing | Row/column clustering |
| Diagonal | Line at 30–60° angle across stage | Angle of linear regression slope |
| Cluster | Grouped near centre with irregular spacing | Low variance in both X and Y |
| Scattered | High variance in both axes, no clear pattern | No template matched |
| Circle / arc | Curved arrangement | Fit to ellipse equation |

The formation label is included in the output and influences how transition movements are suggested (e.g., a line formation transitions naturally to a V by having the centre dancer step forward).

---

### E. Song Structure Analysis

**Method:** BPM detection + section labelling.

If MP3 is uploaded:
- Use `librosa` to compute BPM and detect beat frames.
- Use `librosa.effects.split` to detect energy-based section boundaries.
- Label sections as intro / verse / chorus / bridge / outro based on energy envelope and repetition patterns.

If song name or link is provided:
- Look up known song structure from music knowledge or web search.
- Use BPM and section timestamps from the knowledge base.

**Output: Song Map**

```json
{
  "title": "Shape of You",
  "bpm": 96,
  "duration_s": 234,
  "sections": [
    {"label": "Intro",   "start_s": 0,   "end_s": 15,  "energy": "low",    "rhythm": "smooth"},
    {"label": "Verse 1", "start_s": 15,  "end_s": 47,  "energy": "medium", "rhythm": "groove"},
    {"label": "Pre-Ch",  "start_s": 47,  "end_s": 63,  "energy": "rising", "rhythm": "building"},
    {"label": "Chorus",  "start_s": 63,  "end_s": 95,  "energy": "high",   "rhythm": "driving"},
    {"label": "Verse 2", "start_s": 95,  "end_s": 127, "energy": "medium", "rhythm": "groove"},
    {"label": "Chorus",  "start_s": 127, "end_s": 159, "energy": "high",   "rhythm": "driving"},
    {"label": "Bridge",  "start_s": 159, "end_s": 191, "energy": "peak",   "rhythm": "intense"},
    {"label": "Chorus",  "start_s": 191, "end_s": 223, "energy": "high",   "rhythm": "driving"},
    {"label": "Outro",   "start_s": 223, "end_s": 234, "energy": "fading", "rhythm": "smooth"}
  ]
}
```

---

### F. Movement Plan Generation

**Method:** Rule-based choreography mapping + AI language model reasoning.

Using the dancer positions, formation type, initial postures, and song map, the agent generates a movement plan structured as follows:

#### Movement Vocabulary by Dance Style

| Style | Suggested Movement Types |
|-------|--------------------------|
| Hip-hop | Bounce, isolations, footwork, wave, freeze, floor drop |
| Contemporary | Flow, reach, fall-and-recover, spiral, swing, stillness |
| K-pop | Synchronized arm choreo, point, formation snap, side-step |
| Ballet | Port de bras, tendu, arabesque, promenade, développé |
| Generic/unspecified | Step-touch, open arms, turn, level change, approach/retreat |

#### Formation Transition Rules

| From | To | Trigger | Transition Style |
|------|----|---------|-----------------|
| Line | V-shape | Chorus hit | Centre dancer steps forward on beat 1 |
| Scattered | Grid | Verse start | Dancers walk to grid marks over 8 beats |
| V-shape | Circle | Bridge | Wings arc outward, lead stays centre |
| Any | Freeze | Song pause or silence | All dancers hold last pose |
| Grid | Diagonal | Chorus climax | Left column steps downstage over 4 beats |

#### Timestamped Movement Plan Structure (per dancer, per section)

```json
{
  "dancer_id": "D2",
  "section": "Chorus",
  "start_s": 63,
  "end_s": 95,
  "movements": [
    {"beat": 1,  "time_s": 63.0, "action": "Step right, open arms wide"},
    {"beat": 3,  "time_s": 64.3, "action": "Clap overhead, weight shift left"},
    {"beat": 5,  "time_s": 65.6, "action": "Pivot 180°, face upstage"},
    {"beat": 9,  "time_s": 68.1, "action": "Level drop to crouch"},
    {"beat": 13, "time_s": 70.7, "action": "Rise on beat, arms sweep low to high"},
    {"beat": 17, "time_s": 73.2, "action": "Isolation: chest pop left then right"},
    {"beat": 25, "time_s": 78.3, "action": "Cross stage to position (0.55, 0.40)"},
    {"beat": 33, "time_s": 83.4, "action": "Freeze: hold arabesque-style balance"}
  ],
  "formation_at_section_end": "V-shape"
}
```

---

## 6. Step-by-Step Workflow

```
[USER in Telegram]
  │
  ├── Sends: formation image + song (MP3 / name / link) + optional metadata
  │
  ▼
[STEP 1 — Image Validation]
  │  Check image is readable, not too blurry, and contains at least 1 person.
  │  If blurry: send warning. If no persons detected: ask user to resend a clearer image.
  ▼
[STEP 2 — Image Preprocessing]
  │  CLAHE brightness normalisation → denoising → resolution upscale if needed.
  ▼
[STEP 3 — Person Detection]
  │  YOLOv8 detects all dancers. Assigns IDs D1–DN left to right.
  │  Records bounding boxes and normalised stage coordinates.
  ▼
[STEP 4 — Pose Estimation]
  │  MediaPipe Pose runs on each dancer crop.
  │  Records facing direction and initial posture for each dancer.
  ▼
[STEP 5 — Formation Classification]
  │  Geometric analysis of dancer positions → formation label.
  ▼
[STEP 6 — Song Analysis]
  │  If MP3: librosa BPM + section detection.
  │  If name/link: retrieve known song structure.
  │  Outputs: BPM, duration, section timestamps, energy levels.
  ▼
[STEP 7 — Movement Plan Generation]
  │  Match dance style (from hint or inferred from formation/song genre).
  │  For each song section: assign movements to each dancer at beat-level timestamps.
  │  Plan formation transitions between sections.
  │  Respect initial positions from Step 3.
  ▼
[STEP 8 — Output Generation]
  │  Default: Generate interactive HTML dashboard (see Section 9).
  │  If user requested PDF: convert HTML to PDF via WeasyPrint or Puppeteer.
  ▼
[STEP 9 — Telegram Delivery]
  │  Send HTML file (or PDF) directly to the Telegram chat.
  │  Include a plain-text summary message with key facts:
  │    – Number of dancers detected
  │    – Song BPM and duration
  │    – Formation detected
  │    – Number of movement cues generated
  └── Done.
```

---

## 7. Output Format

### A. Plain-Text Summary (sent with file)

```
✅ DanceFlow Plan Ready

🎵 Song: Shape of You – Ed Sheeran (BPM: 96, Duration: 3:54)
👥 Dancers detected: 5 (D1–D5)
🗺 Starting formation: Diagonal line
📋 Movement cues generated: 143 across 9 sections

Your choreography dashboard is attached.
Open the HTML file in any browser, or request PDF with /pdf
```

---

### B. HTML Dashboard

The HTML dashboard is a self-contained single file with no external dependencies.

#### Dashboard Sections

**Section 1 — Stage Map (Formation View)**
- SVG top-down view of the stage floor (rectangle).
- Each dancer shown as a labelled dot (D1–D5) at their detected starting position.
- Formation outline drawn as a connecting shape (line, V, circle, etc.).
- Click a dancer dot to highlight their movements in the timeline below.

**Section 2 — Song Timeline Bar**
- Horizontal bar spanning full song duration.
- Colour-coded sections: intro (grey), verse (blue), chorus (orange), bridge (red), outro (grey).
- Beat markers drawn as tick marks.
- Hover over any section to see energy level and rhythm character.

**Section 3 — Movement Timeline Grid**
- Rows: one per dancer (D1 to DN).
- Columns: song sections with timestamp labels.
- Each cell contains the key movement cues for that dancer in that section.
- Cells colour-coded by movement energy: low (light blue), medium (teal), high (orange), climax (red).
- Click any cell to expand movement detail (beat-by-beat list).

**Section 4 — Formation Transition Cards**
- One card per section boundary where formation changes.
- Shows: before-formation → after-formation, which dancers move, timestamp, direction of travel.
- Each card has a mini before/after SVG diagram.

**Section 5 — Export Controls**
- `Download PDF` button — triggers PDF export via browser print (`@media print`).
- `Copy Full Plan` button — copies the complete movement plan as plain text to clipboard.
- `Reset Filters` button — clears any dancer or section filters.

#### HTML Template Structure

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>DanceFlow — Choreography Plan</title>
  <style>
    :root {
      --color-bg: #F9F7F4;
      --color-surface: #FFFFFF;
      --color-border: #E2DDD7;
      --color-accent: #D4783A;
      --color-low: #C8DCF0;
      --color-medium: #5B9BD5;
      --color-high: #E07B39;
      --color-climax: #C0392B;
      --color-text: #2C2C2C;
      --color-muted: #8A8A8A;
      --font-main: system-ui, -apple-system, sans-serif;
    }

    body {
      font-family: var(--font-main);
      background: var(--color-bg);
      margin: 0;
      padding: 24px;
      color: var(--color-text);
    }

    h1 { font-size: 22px; margin-bottom: 4px; }
    .subtitle { font-size: 13px; color: var(--color-muted); margin-bottom: 32px; }

    /* Stage Map */
    .stage-container {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: 12px;
      padding: 20px;
      margin-bottom: 28px;
    }
    .stage-svg { width: 100%; max-width: 600px; display: block; margin: 0 auto; }

    /* Song Timeline */
    .timeline-bar { display: flex; height: 36px; border-radius: 8px; overflow: hidden;
                    margin-bottom: 8px; }
    .section-block { display: flex; align-items: center; justify-content: center;
                     font-size: 11px; font-weight: 600; color: #fff;
                     cursor: pointer; transition: opacity 0.2s; }
    .section-block:hover { opacity: 0.8; }

    /* Movement Grid */
    .grid-table { width: 100%; border-collapse: collapse; font-size: 12px; }
    .grid-table th { background: #3D3D3D; color: #fff; padding: 8px 12px;
                     text-align: left; font-weight: 600; }
    .grid-table td { border: 1px solid var(--color-border); padding: 10px 12px;
                     vertical-align: top; cursor: pointer; }
    .grid-table tr:hover td { background: #FFF8F2; }
    .energy-low    { border-left: 4px solid var(--color-low); }
    .energy-medium { border-left: 4px solid var(--color-medium); }
    .energy-high   { border-left: 4px solid var(--color-high); }
    .energy-climax { border-left: 4px solid var(--color-climax); }

    .movement-cue { display: block; margin-bottom: 3px; }
    .beat-tag { font-size: 10px; color: var(--color-muted);
                font-weight: 700; margin-right: 4px; }

    /* Formation Cards */
    .formation-cards { display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 28px; }
    .formation-card {
      background: var(--color-surface);
      border: 1px solid var(--color-border);
      border-radius: 10px;
      padding: 16px;
      min-width: 200px;
      flex: 1;
    }
    .formation-card h4 { margin: 0 0 8px; font-size: 13px; }
    .formation-before-after { display: flex; align-items: center; gap: 8px; }

    /* Buttons */
    .btn {
      padding: 10px 20px; border: none; border-radius: 8px;
      cursor: pointer; font-size: 13px; font-weight: 600;
      margin-right: 10px; margin-top: 16px;
    }
    .btn-primary { background: #3D3D3D; color: #fff; }
    .btn-primary:hover { background: #555; }
    .btn-secondary { background: var(--color-border); color: var(--color-text); }

    @media print {
      .btn { display: none; }
      body { padding: 10px; }
    }
  </style>
</head>
<body>

<h1>🎵 DanceFlow — Choreography Plan</h1>
<p class="subtitle" id="subtitle-line">Loading plan…</p>

<!-- Stage Map -->
<div class="stage-container">
  <h3 style="margin-top:0">Starting Formation</h3>
  <svg class="stage-svg" viewBox="0 0 500 300"
       xmlns="http://www.w3.org/2000/svg" id="stage-svg">
    <!-- Injected by buildStageMap() -->
  </svg>
</div>

<!-- Song Timeline -->
<div class="stage-container">
  <h3 style="margin-top:0">Song Timeline</h3>
  <div class="timeline-bar" id="timeline-bar"></div>
  <div id="timeline-labels" style="display:flex; font-size:10px;
       color:var(--color-muted); margin-top:4px;"></div>
</div>

<!-- Movement Grid -->
<div class="stage-container">
  <h3 style="margin-top:0">Movement Plan by Dancer</h3>
  <table class="grid-table" id="movement-grid">
    <!-- Injected by buildMovementGrid() -->
  </table>
</div>

<!-- Formation Transitions -->
<div class="stage-container">
  <h3 style="margin-top:0">Formation Transitions</h3>
  <div class="formation-cards" id="formation-cards">
    <!-- Injected by buildFormationCards() -->
  </div>
</div>

<!-- Export Controls -->
<button class="btn btn-primary" onclick="window.print()">Download PDF</button>
<button class="btn btn-secondary" onclick="copyPlan()">Copy Full Plan</button>

<script>
// ── DATA INJECTION ──────────────────────────────────────────────────────────
// The following constants are injected by the agent at generation time.

const SONG_INFO = {
  title: "Song Title",
  artist: "Artist Name",
  bpm: 96,
  duration_s: 234,
  sections: [
    // { label, start_s, end_s, energy, rhythm }
  ]
};

const DANCERS = [
  // { id, label, x_norm, y_norm, facing, posture }
];

const MOVEMENT_PLAN = [
  // { dancer_id, section, start_s, end_s, movements: [{beat, time_s, action}] }
];

const FORMATION_TRANSITIONS = [
  // { at_time_s, at_section, from_formation, to_formation, description }
];

// ── SECTION COLOURS ─────────────────────────────────────────────────────────
const SECTION_COLORS = {
  "Intro":   "#9E9E9E",
  "Verse 1": "#5B9BD5",
  "Verse 2": "#5B9BD5",
  "Pre-Ch":  "#7B68EE",
  "Chorus":  "#E07B39",
  "Bridge":  "#C0392B",
  "Outro":   "#9E9E9E"
};

const ENERGY_CLASS = {
  "low": "energy-low",
  "medium": "energy-medium",
  "high": "energy-high",
  "peak": "energy-climax",
  "rising": "energy-medium",
  "fading": "energy-low"
};

// ── STAGE MAP ────────────────────────────────────────────────────────────────
function buildStageMap() {
  const svg = document.getElementById('stage-svg');
  const W = 500, H = 300, PAD = 30;

  // Stage border
  svg.innerHTML = `
    <rect x="${PAD}" y="${PAD}" width="${W - PAD*2}" height="${H - PAD*2}"
          fill="#F0ECE6" stroke="#C0BAB3" stroke-width="2" rx="4"/>
    <text x="${W/2}" y="${H - 8}" text-anchor="middle"
          font-size="11" fill="#C0BAB3">STAGE FRONT</text>`;

  DANCERS.forEach(d => {
    const px = PAD + d.x_norm * (W - PAD*2);
    const py = PAD + d.y_norm * (H - PAD*2);

    svg.innerHTML += `
      <circle cx="${px}" cy="${py}" r="16"
              fill="#3D3D3D" stroke="#fff" stroke-width="2"
              style="cursor:pointer"
              onclick="highlightDancer('${d.id}')"/>
      <text x="${px}" y="${py + 4}" text-anchor="middle"
            font-size="11" font-weight="700" fill="#fff">${d.id}</text>`;
  });
}

// ── TIMELINE BAR ─────────────────────────────────────────────────────────────
function buildTimeline() {
  const bar = document.getElementById('timeline-bar');
  const labels = document.getElementById('timeline-labels');
  const total = SONG_INFO.duration_s;

  SONG_INFO.sections.forEach(s => {
    const pct = ((s.end_s - s.start_s) / total * 100).toFixed(1);
    const col = SECTION_COLORS[s.label] || "#888";
    bar.innerHTML += `
      <div class="section-block"
           style="width:${pct}%;background:${col}"
           title="${s.label}: ${s.start_s}s–${s.end_s}s · ${s.energy} energy">
        ${pct > 6 ? s.label : ''}
      </div>`;
    labels.innerHTML += `
      <div style="width:${pct}%;text-align:center;overflow:hidden">
        ${s.start_s}s
      </div>`;
  });
}

// ── MOVEMENT GRID ─────────────────────────────────────────────────────────────
function buildMovementGrid() {
  const table = document.getElementById('movement-grid');
  const sections = SONG_INFO.sections.map(s => s.label);

  // Header row
  let headerHTML = '<tr><th>Dancer</th>';
  sections.forEach(s => { headerHTML += `<th>${s}</th>`; });
  headerHTML += '</tr>';
  table.innerHTML = headerHTML;

  // One row per dancer
  DANCERS.forEach(dancer => {
    let rowHTML = `<tr><td><strong>${dancer.id}</strong>
      ${dancer.label ? `<br><span style="font-size:11px;color:#888">${dancer.label}</span>` : ''}
      </td>`;

    sections.forEach(sectionLabel => {
      const sectionInfo = SONG_INFO.sections.find(s => s.label === sectionLabel);
      const plan = MOVEMENT_PLAN.find(
        p => p.dancer_id === dancer.id && p.section === sectionLabel
      );
      const energyClass = ENERGY_CLASS[sectionInfo ? sectionInfo.energy : "low"] || "";

      if (plan && plan.movements.length > 0) {
        let cueHTML = plan.movements.slice(0, 4).map(m =>
          `<span class="movement-cue">
            <span class="beat-tag">B${m.beat}</span>${m.action}
          </span>`
        ).join('');
        if (plan.movements.length > 4) {
          cueHTML += `<span style="font-size:10px;color:#888">
            +${plan.movements.length - 4} more…</span>`;
        }
        rowHTML += `<td class="${energyClass}" title="Click for full detail"
                       onclick="showDetail('${dancer.id}','${sectionLabel}')">
                      ${cueHTML}</td>`;
      } else {
        rowHTML += `<td style="color:#ccc;font-size:11px">—</td>`;
      }
    });

    rowHTML += '</tr>';
    table.innerHTML += rowHTML;
  });
}

// ── FORMATION CARDS ───────────────────────────────────────────────────────────
function buildFormationCards() {
  const container = document.getElementById('formation-cards');
  FORMATION_TRANSITIONS.forEach(t => {
    container.innerHTML += `
      <div class="formation-card">
        <h4>At ${t.at_time_s}s — ${t.at_section}</h4>
        <div class="formation-before-after">
          <span style="font-size:12px;background:#eee;padding:4px 8px;
                border-radius:4px">${t.from_formation}</span>
          <span style="font-size:18px;color:#D4783A">→</span>
          <span style="font-size:12px;background:#3D3D3D;color:#fff;
                padding:4px 8px;border-radius:4px">${t.to_formation}</span>
        </div>
        <p style="font-size:11px;color:#888;margin:8px 0 0">${t.description}</p>
      </div>`;
  });
}

// ── DETAIL MODAL ──────────────────────────────────────────────────────────────
function showDetail(dancerId, sectionLabel) {
  const plan = MOVEMENT_PLAN.find(
    p => p.dancer_id === dancerId && p.section === sectionLabel
  );
  if (!plan) return;
  const lines = plan.movements.map(
    m => `Beat ${m.beat} (${m.time_s}s): ${m.action}`
  ).join('\n');
  alert(`${dancerId} — ${sectionLabel}\n\n${lines}`);
}

// ── HIGHLIGHT ─────────────────────────────────────────────────────────────────
function highlightDancer(id) {
  document.querySelectorAll('#movement-grid tr').forEach(row => {
    row.style.opacity = row.querySelector('td')
      && row.querySelector('td').textContent.trim().startsWith(id)
      ? '1' : '0.3';
  });
}

// ── COPY PLAN ─────────────────────────────────────────────────────────────────
function copyPlan() {
  let text = `DanceFlow Plan — ${SONG_INFO.title} by ${SONG_INFO.artist}\n`;
  text += `BPM: ${SONG_INFO.bpm} | Duration: ${SONG_INFO.duration_s}s\n\n`;
  MOVEMENT_PLAN.forEach(p => {
    text += `${p.dancer_id} — ${p.section} (${p.start_s}s–${p.end_s}s)\n`;
    p.movements.forEach(m => {
      text += `  Beat ${m.beat} (${m.time_s}s): ${m.action}\n`;
    });
    text += '\n';
  });
  navigator.clipboard.writeText(text).then(
    () => alert('Plan copied to clipboard!'),
    () => alert('Copy failed. Please select and copy manually.')
  );
}

// ── SUBTITLE ──────────────────────────────────────────────────────────────────
function buildSubtitle() {
  document.getElementById('subtitle-line').textContent =
    `${SONG_INFO.title} · ${SONG_INFO.artist} · BPM ${SONG_INFO.bpm} · ` +
    `${DANCERS.length} dancers · Generated by DanceFlow v1.0`;
}

// ── INIT ──────────────────────────────────────────────────────────────────────
buildSubtitle();
buildStageMap();
buildTimeline();
buildMovementGrid();
buildFormationCards();
</script>
</body>
</html>
```

---

### C. PDF Output

When the user requests PDF (by sending `/pdf` or including `"PDF"` in their message):

- The same HTML template is rendered headlessly via **Puppeteer** (Node.js) or **WeasyPrint** (Python).
- PDF is generated with A3 landscape orientation to accommodate the wide movement grid.
- Page breaks are inserted between the stage map, timeline, movement grid, and formation cards.
- The PDF is sent directly as a Telegram document attachment.

```python
# PDF generation via WeasyPrint
from weasyprint import HTML, CSS

def export_pdf(html_content, output_path):
    css = CSS(string="""
        @page { size: A3 landscape; margin: 15mm; }
        .btn { display: none !important; }
        body { padding: 0; }
    """)
    HTML(string=html_content).write_pdf(output_path, stylesheets=[css])
```

---

## 8. Skill File Structure (Hermes / OpenClaw)

```yaml
skill_name: DanceFlow
version: 1.0
target_user: Dance instructors, choreographers, student dance groups, performing arts students
real_world_problem: >
  Planning timestamped movements for multiple dancers across a whole song is
  time-consuming and hard to communicate. DanceFlow automates this by detecting
  dancers in a formation photo and mapping movements to song sections.

input_format:
  image:
    type: still photo
    format: [JPG, PNG, WebP]
    content: all dancers in starting formation
  song:
    options:
      - MP3 file upload
      - song name as text
      - YouTube / Spotify / SoundCloud URL
  optional:
    dance_style: hint for movement vocabulary
    dancer_names: mapping of IDs to real names
    stage_dimensions: in metres
    output_format: HTML (default) or PDF

cv_methods:
  - Image preprocessing (CLAHE, denoising, blur detection)
  - Person detection (YOLOv8, COCO pretrained)
  - Pose estimation (MediaPipe Pose, per-dancer crop)
  - Spatial layout / formation classification (geometric clustering)
  - Song structure analysis (librosa BPM + section detection)

workflow:
  1: Validate and preprocess the formation image
  2: Detect all dancers using YOLOv8; assign IDs left-to-right
  3: Run MediaPipe Pose per dancer crop; record posture and facing
  4: Classify starting formation from dancer coordinates
  5: Analyse song for BPM, section timestamps, and energy levels
  6: Generate timestamped movement plan per dancer per section
  7: Build HTML dashboard or PDF
  8: Send output file + summary text to Telegram

output_format:
  default: interactive HTML dashboard (self-contained, no dependencies)
  on_request: PDF (A3 landscape, printable)
  summary_text: dancer count, BPM, formation, cue count

limitation_handling:
  blurry_image: warn user, attempt anyway, flag low-confidence detections
  no_person_found: ask user to resend clearer full-body photo
  unknown_song: use tempo and genre estimate; warn that structure is approximate
  occluded_dancer: note partial detection; label as DX_partial
  audio_upload_fail: fall back to song name lookup
  more_than_10_dancers: cap at 10; warn user; suggest splitting into sub-groups

ethical_boundary:
  - System comments only on stage positions, movement timing, and formation design.
  - System does NOT evaluate physical appearance, attractiveness, or body type.
  - System does NOT identify individuals by face or personal data.
  - System does NOT store any uploaded images or audio beyond the active session.
  - Dancer labels are positional (D1–DN) unless the user explicitly provides names.
  - All generated movement plans are suggestions, not prescriptions.
    The choreographer retains full creative authority.
```

---

## 9. Telegram Gateway Integration

### Trigger Commands

| Command | Action |
|---------|--------|
| `/start` | Welcome message + instructions |
| `/plan` | Start a new choreography plan (prompts for image and song) |
| `/pdf` | Re-export the last plan as PDF |
| `/help` | List all commands and input formats |

### Conversation Flow

```
User: /plan
Bot:  👋 Send me a photo of your dancers in their starting positions,
      then add the song name, a Spotify/YouTube link, or upload an MP3.
      You can also tell me the dance style (e.g. hip-hop, K-pop, contemporary).

User: [sends photo] Shape of You - Ed Sheeran, hip-hop style, 5 dancers = Ana, Ben, Cara, Dev, Eva

Bot:  🔍 Analysing formation image…
      ✅ 5 dancers detected (D1–D5, left to right: Ana, Ben, Cara, Dev, Eva)
      📐 Starting formation: Diagonal line
      🎵 Song: Shape of You – Ed Sheeran (BPM: 96, 3:54)
      ⚙️ Generating choreography plan…

Bot:  ✅ DanceFlow Plan Ready!

      🎵 Shape of You – Ed Sheeran · BPM 96 · 3:54
      👥 5 dancers · Diagonal line starting formation
      📋 147 movement cues across 9 sections
      💃 Style: Hip-hop

      Open the attached HTML dashboard in any browser.
      For a printable PDF, send /pdf

Bot:  [sends choreography_plan.html]
```

---

## 10. Limitation and Ethical Awareness

### Known Technical Limitations

| Limitation | Cause | Mitigation |
|------------|-------|-----------|
| Occlusion in formation photo | Dancers standing behind others | Warn user; detect visible dancers only; suggest taking photo from elevated angle |
| Blurry or dark images | Poor lighting in rehearsal space | Preprocessing corrects mild issues; severe blur triggers re-upload request |
| Pose estimation inaccuracy | Loose or layered clothing hides body contours | Confidence score shown; user can override posture manually |
| Unknown song structure | Obscure or unreleased tracks | BPM estimated from audio; section labels approximate; user can provide section timestamps |
| Generated movements are suggestions | AI cannot choreograph perfectly | Always framed as a starting plan; choreographer must review and adjust |
| MP3 audio upload size | Large files slow processing | Recommend sending first 60s clip or using song name instead |
| More than 10 dancers | High complexity | Cap at 10 detected dancers; suggest splitting into groups |

### Ethical Boundaries

- **No facial identification.** The system uses bounding boxes and positions only. It never attempts to identify who a dancer is from their face.
- **No appearance judgements.** The system describes positions, orientations, and movement sequences. It never comments on body type, attractiveness, weight, or physical characteristics.
- **No biometric storage.** No image data, pose data, or audio is stored beyond the active session.
- **Creative authority remains with the choreographer.** All output is labelled as AI-generated suggestions. The system does not claim that any movement plan is artistically correct or final.
- **No competitive assessment.** The system does not rank or judge dancers' abilities relative to one another.

---

## 11. What Makes a Strong DanceFlow Output?

A strong output follows this structure:

```
Real user (choreographer / instructor)
  → real visual problem (tracking multiple dancers across a full song)
    → suitable CV technique (person detection + pose estimation + spatial analysis)
      → structured output (timestamped movement plan per dancer)
        → useful decision support (choreographer can walk into rehearsal with a ready plan)
```

The output should not only say:

> I detected 5 dancers.

It should help the choreographer answer:

> At 1:03, when the chorus hits, where should Ana be standing, and what should she do with her arms?

That is the standard expected for this skill.
