---
name: danceflow
description: "Accept a still image of all dancers in their starting formation + a song (MP3, name, or link) and produce a full timestamped choreography movement plan for every dancer. Primary output is a PDF (sent directly via Telegram) with embedded formation diagrams for every transition. A secondary interactive HTML dashboard with animated formation transitions is available on request via /web."
tags: [dance, choreography, pose-estimation, person-detection, rhythm-analysis, movement-planning, telegram, pdf, animation]
version: 2.0
---

# DanceFlow — Formation-to-Choreography Planner

## 1. Skill Name

**DanceFlow** — an AI agent skill that detects dancers from a formation photo, analyses a song's structure and rhythm, and generates a complete timestamped choreography plan. Delivered as a **PDF directly in Telegram** with embedded formation diagrams, or as an **interactive animated HTML dashboard** on request.

---

## 2. Target User

| User Type | Use Case |
|-----------|----------|
| Dance instructors / choreographers | Plan full-song movement sequences before rehearsal |
| Student dance groups | Get an AI-assisted plan for competitions or events |
| Performing arts students | Understand how rhythm maps to movement |
| Event and production managers | Review and plan stage usage for dance acts |

**Primary target:** Non-technical dance practitioners who need a ready-to-use choreography plan without notation training or specialist software.

---

## 3. Real-World Problem

Planning group choreography requires tracking every dancer's position, matching movements to song sections, and communicating the plan clearly to all dancers. Without a structured plan, rehearsals are slow and movements are easily forgotten.

**DanceFlow** solves this by detecting all dancers from a single photo, analysing the song, generating a beat-level movement plan, and delivering it as a visual PDF or animated dashboard.

---

## 4. Input Format

| Input | Format | Notes |
|-------|--------|-------|
| Formation image | JPG, PNG, WebP | Full-body shot of all dancers in starting positions; frontal or elevated view preferred |
| Song | MP3 upload, song name, or URL | YouTube / Spotify / SoundCloud accepted |
| Dance style *(optional)* | Text | e.g. `hip-hop`, `K-pop`, `contemporary` — adjusts movement vocabulary |
| Dancer names *(optional)* | Text | e.g. `D1=Ana, D2=Ben` — replaces numbered labels |
| Stage size *(optional)* | Text | e.g. `8m x 6m` — scales the position grid |
| Output format *(optional)* | `/web` command | Requests interactive HTML instead of default PDF |

---

## 5. CV / Image-Processing Method

### A. Image Preprocessing

| Step | Method | Purpose |
|------|--------|---------|
| Blur detection | Laplacian variance | Warn if score < 100; request clearer photo |
| Brightness normalisation | CLAHE (LAB colour space) | Fix underexposed rehearsal photos |
| Denoising | `fastNlMeansDenoising` | Remove grain and compression artefacts |
| Resolution upscale | Bicubic interpolation | Ensure minimum 640px height for detection |

```python
import cv2

def preprocess(image_path):
    img = cv2.imread(image_path)
    # Blur check
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < 100:
        print(f"WARNING: Blurry image (score={blur_score:.1f})")
    # Brightness normalisation via CLAHE
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(l)
    img = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # Denoise
    img = cv2.fastNlMeansDenoisingColored(img, h=10)
    # Upscale if needed
    h, w = img.shape[:2]
    if h < 640:
        img = cv2.resize(img, (int(w * 640/h), 640), interpolation=cv2.INTER_CUBIC)
    return img, blur_score
```

### B. Person Detection

**Method:** YOLOv8 pretrained (COCO). Detects all `person` class bounding boxes with confidence > 0.5. Dancers sorted left-to-right and labelled `D1`–`DN`. Normalised stage coordinates `(x_norm, y_norm)` recorded per dancer.

```python
from ultralytics import YOLO
model = YOLO("yolov8n.pt")

def detect_dancers(img):
    results = model(img)
    dancers = []
    for box in results[0].boxes:
        if int(box.cls) == 0 and float(box.conf) > 0.5:
            cx = float((box.xyxy[0][0] + box.xyxy[0][2]) / 2)
            cy = float((box.xyxy[0][1] + box.xyxy[0][3]) / 2)
            dancers.append({"cx": cx, "cy": cy})
    dancers.sort(key=lambda d: d["cx"])
    h, w = img.shape[:2]
    for i, d in enumerate(dancers):
        d["id"] = f"D{i+1}"
        d["x_norm"] = round(d["cx"] / w, 3)
        d["y_norm"] = round(d["cy"] / h, 3)
    return dancers
```

### C. Pose Estimation

**Method:** MediaPipe Pose on each dancer's cropped bounding box. Extracts facing direction (front/side/back) and initial posture (standing, crouched, arms raised, etc.) from landmark visibility scores.

### D. Formation Classification

**Method:** Geometric analysis of `(x_norm, y_norm)` coordinates matched against formation templates.

| Formation | Detection Logic |
|-----------|----------------|
| Straight line | Linear regression R² > 0.95 |
| V-shape | Triangle vertex detection — one dancer forward |
| Grid | Row/column clustering with equal spacing |
| Diagonal | Regression slope 30–60° |
| Circle / arc | Fit to ellipse equation |
| Scattered | No template matched (high variance both axes) |

### E. Song Structure Analysis

**Method:** `librosa` for MP3 uploads (BPM + energy-based section boundaries). Known song structure lookup for named songs or links.

**Output per song:**
```json
{
  "title": "Shape of You", "artist": "Ed Sheeran",
  "bpm": 96, "duration_s": 234,
  "sections": [
    {"label": "Intro",   "start_s": 0,   "end_s": 15,  "energy": "low"},
    {"label": "Verse 1", "start_s": 15,  "end_s": 47,  "energy": "medium"},
    {"label": "Chorus",  "start_s": 63,  "end_s": 95,  "energy": "high"},
    {"label": "Bridge",  "start_s": 159, "end_s": 191, "energy": "peak"},
    {"label": "Outro",   "start_s": 223, "end_s": 234, "energy": "low"}
  ]
}
```

---

## 6. Step-by-Step Workflow

```
USER sends image + song via Telegram
  │
  ├─ STEP 1  Validate image (blur check, person present?)
  ├─ STEP 2  Preprocess image (CLAHE, denoise, upscale)
  ├─ STEP 3  Detect dancers → assign D1–DN with (x_norm, y_norm)
  ├─ STEP 4  Pose estimation per dancer → posture + facing
  ├─ STEP 5  Classify starting formation
  ├─ STEP 6  Analyse song → BPM, sections, energy levels
  ├─ STEP 7  Generate movement plan (beat-level, per dancer, per section)
  ├─ STEP 8  Compute formation state per section → (x_norm, y_norm) per dancer
  │
  ├─ DEFAULT OUTPUT
  │    Render formation diagrams as SVG → rasterise to PNG via cairosvg
  │    Build PDF (WeasyPrint): cover + section pages + transition pages
  │    Send PDF to Telegram chat
  │
  └─ IF /web requested
       Build self-contained HTML dashboard with animated SVG stage
       Send HTML file to Telegram chat
```

---

## 7. Output Format

### A. Telegram Summary Message (always sent first)

```
✅ DanceFlow Plan Ready

🎵 Shape of You – Ed Sheeran  |  BPM: 96  |  3:54
👥 5 dancers detected  |  Starting: Diagonal line
📋 147 movement cues across 9 sections

📄 Choreography PDF attached.
Type /web for the interactive animated dashboard.
```

---

### B. Primary Output — PDF (sent directly in Telegram)

Generated via **WeasyPrint** (Python). A3 landscape. Sent as `choreography_plan.pdf`.

#### Page Structure

| Page | Content |
|------|---------|
| 1 — Cover | Song title, artist, BPM, duration, dancer count, starting formation name, starting formation diagram |
| 2 — Song Overview | Colour-coded section timeline bar + energy level per section |
| 3…N — Section Pages | One page per song section: movement table (dancer rows × beat columns) + formation diagram for that section |
| N+1…end — Transition Pages | One page per formation change: **before formation plot → after formation plot** side by side with dancer-labelled dots, timestamp, and section label |

#### Formation Diagram Specification (embedded in PDF)

Each formation diagram is a top-down SVG stage view rendered to PNG via `cairosvg`:

- Rectangle = stage floor (proportional to stage dimensions if provided).
- Filled circle per dancer, labelled with ID or name.
- Connecting lines drawn to indicate the formation shape (line, V, grid, circle, etc.).
- Colour coding: current section = filled dark circles; previous section = ghost circles (grey, lower opacity) for before/after comparison on transition pages.

```python
import cairosvg

def render_formation_svg(dancers, stage_w=500, stage_h=300,
                          ghost_dancers=None, label="Formation"):
    """
    dancers:       list of {id, x_norm, y_norm, name}
    ghost_dancers: list of same structure for before-state (shown as grey ghosts)
    Returns PNG bytes for embedding in PDF.
    """
    PAD = 40
    circles = ""

    # Ghost positions (before-state)
    if ghost_dancers:
        for d in ghost_dancers:
            px = PAD + d["x_norm"] * (stage_w - PAD*2)
            py = PAD + d["y_norm"] * (stage_h - PAD*2)
            circles += f'''
              <circle cx="{px}" cy="{py}" r="18"
                      fill="#CCCCCC" opacity="0.5"/>
              <text x="{px}" y="{py+4}" text-anchor="middle"
                    font-size="11" fill="#999">{d["id"]}</text>'''

    # Arrows from ghost to current (transition only)
    if ghost_dancers:
        for g, c in zip(ghost_dancers, dancers):
            gx = PAD + g["x_norm"] * (stage_w - PAD*2)
            gy = PAD + g["y_norm"] * (stage_h - PAD*2)
            cx2 = PAD + c["x_norm"] * (stage_w - PAD*2)
            cy2 = PAD + c["y_norm"] * (stage_h - PAD*2)
            circles += f'''
              <line x1="{gx}" y1="{gy}" x2="{cx2}" y2="{cy2}"
                    stroke="#D4783A" stroke-width="1.5"
                    stroke-dasharray="4,3" marker-end="url(#arr)"/>'''

    # Current positions
    for d in dancers:
        px = PAD + d["x_norm"] * (stage_w - PAD*2)
        py = PAD + d["y_norm"] * (stage_h - PAD*2)
        name = d.get("name", d["id"])
        circles += f'''
          <circle cx="{px}" cy="{py}" r="18"
                  fill="#3D3D3D" stroke="#fff" stroke-width="2"/>
          <text x="{px}" y="{py+4}" text-anchor="middle"
                font-size="11" font-weight="700" fill="#fff">{name}</text>'''

    svg = f'''<svg width="{stage_w}" height="{stage_h}"
                   xmlns="http://www.w3.org/2000/svg">
      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6"
                refX="6" refY="3" orient="auto">
          <path d="M0,0 L6,3 L0,6 Z" fill="#D4783A"/>
        </marker>
      </defs>
      <rect x="{PAD}" y="{PAD}"
            width="{stage_w-PAD*2}" height="{stage_h-PAD*2}"
            fill="#F5F2EE" stroke="#C0BAB3" stroke-width="2" rx="4"/>
      <text x="{stage_w/2}" y="{stage_h-10}" text-anchor="middle"
            font-size="11" fill="#C0BAB3">STAGE FRONT</text>
      <text x="{stage_w/2}" y="24" text-anchor="middle"
            font-size="12" font-weight="600" fill="#3D3D3D">{label}</text>
      {circles}
    </svg>'''

    return cairosvg.svg2png(bytestring=svg.encode())
```

#### PDF Generation

```python
from weasyprint import HTML, CSS

def build_pdf(song_info, dancers, movement_plan, formation_states, output_path):
    """
    formation_states: list of {section_label, dancers: [{id, x_norm, y_norm}]}
    """
    pages_html = []

    # Cover page
    start_png = render_formation_svg(
        formation_states[0]["dancers"], label="Starting Formation"
    )
    cover_img_b64 = base64.b64encode(start_png).decode()
    pages_html.append(f"""
      <div class="page cover">
        <h1>DanceFlow Choreography Plan</h1>
        <h2>{song_info['title']} — {song_info['artist']}</h2>
        <p>BPM: {song_info['bpm']} &nbsp;|&nbsp; Duration: {song_info['duration_s']}s
           &nbsp;|&nbsp; {len(dancers)} dancers</p>
        <img src="data:image/png;base64,{cover_img_b64}" class="formation-img"/>
      </div>""")

    # One page per section
    for i, section in enumerate(song_info["sections"]):
        state = next((s for s in formation_states
                      if s["section_label"] == section["label"]), None)
        formation_png = render_formation_svg(
            state["dancers"] if state else dancers,
            label=f"{section['label']} Formation"
        ) if state else b""
        img_b64 = base64.b64encode(formation_png).decode() if formation_png else ""

        # Movement table rows
        rows = ""
        for dancer in dancers:
            plan = next((p for p in movement_plan
                         if p["dancer_id"] == dancer["id"]
                         and p["section"] == section["label"]), None)
            cues = ""
            if plan:
                cues = " &nbsp;·&nbsp; ".join(
                    f"<b>B{m['beat']}</b> {m['action']}"
                    for m in plan["movements"]
                )
            rows += f"""
              <tr>
                <td class="dancer-cell">{dancer.get('name', dancer['id'])}</td>
                <td>{cues or "—"}</td>
              </tr>"""

        pages_html.append(f"""
          <div class="page section-page">
            <div class="section-header energy-{section['energy']}">
              {section['label']}
              <span class="ts">{section['start_s']}s – {section['end_s']}s</span>
            </div>
            <div class="section-body">
              <table class="movement-table">{rows}</table>
              {"<img src='data:image/png;base64," + img_b64 + "' class='formation-img'/>" if img_b64 else ""}
            </div>
          </div>""")

    # Transition pages (before → after)
    for i in range(1, len(formation_states)):
        prev = formation_states[i-1]
        curr = formation_states[i]
        before_after_png = render_formation_svg(
            curr["dancers"],
            ghost_dancers=prev["dancers"],
            label=f"{prev['section_label']} → {curr['section_label']}"
        )
        ba_b64 = base64.b64encode(before_after_png).decode()
        section_info = next((s for s in song_info["sections"]
                              if s["label"] == curr["section_label"]), {})
        pages_html.append(f"""
          <div class="page transition-page">
            <h3>Formation Transition</h3>
            <p class="ts-label">At {section_info.get('start_s','?')}s —
               {prev['section_label']} → {curr['section_label']}</p>
            <img src="data:image/png;base64,{ba_b64}" class="formation-img-large"/>
            <p class="legend">
              <span class="ghost-dot"></span> Previous positions &nbsp;&nbsp;
              <span class="live-dot"></span> New positions &nbsp;&nbsp;
              <span class="arrow-line">- - ▶</span> Movement path
            </p>
          </div>""")

    css = CSS(string="""
      @page { size: A3 landscape; margin: 15mm; }
      body { font-family: system-ui, sans-serif; color: #2C2C2C; margin: 0; }
      .page { page-break-after: always; padding: 10px; }
      .cover { text-align: center; padding-top: 40px; }
      .cover h1 { font-size: 28px; margin-bottom: 8px; }
      .cover h2 { font-size: 20px; color: #666; font-weight: 400; }
      .section-header { font-size: 18px; font-weight: 700; padding: 8px 14px;
                        border-radius: 6px; margin-bottom: 14px; color: #fff; }
      .energy-low    { background: #9E9E9E; }
      .energy-medium { background: #5B9BD5; }
      .energy-high   { background: #E07B39; }
      .energy-peak   { background: #C0392B; }
      .energy-rising { background: #7B68EE; }
      .energy-fading { background: #9E9E9E; }
      .section-body { display: flex; gap: 20px; align-items: flex-start; }
      .movement-table { flex: 1; border-collapse: collapse; font-size: 12px; }
      .movement-table td { border: 1px solid #E2DDD7; padding: 7px 10px;
                           vertical-align: top; }
      .dancer-cell { font-weight: 700; white-space: nowrap; width: 80px;
                     background: #F5F2EE; }
      .formation-img { width: 320px; border: 1px solid #E2DDD7;
                       border-radius: 8px; }
      .formation-img-large { width: 100%; max-width: 700px; display: block;
                              margin: 16px auto;
                              border: 1px solid #E2DDD7; border-radius: 8px; }
      .ts { font-size: 13px; font-weight: 400; margin-left: 12px; opacity: 0.85; }
      .ts-label { font-size: 14px; color: #666; margin: 4px 0 16px; }
      .transition-page { text-align: center; }
      .transition-page h3 { font-size: 20px; margin-bottom: 4px; }
      .legend { font-size: 12px; color: #888; margin-top: 10px; }
      .ghost-dot { display: inline-block; width: 12px; height: 12px;
                   border-radius: 50%; background: #CCC; opacity: 0.5; }
      .live-dot  { display: inline-block; width: 12px; height: 12px;
                   border-radius: 50%; background: #3D3D3D; }
      .arrow-line { color: #D4783A; font-weight: 700; }
    """)

    full_html = f"<html><body>{''.join(pages_html)}</body></html>"
    HTML(string=full_html).write_pdf(output_path, stylesheets=[css])
```

#### Telegram Send

```python
import telebot

def send_pdf_to_telegram(bot, chat_id, pdf_path, summary_text):
    bot.send_message(chat_id, summary_text)
    with open(pdf_path, "rb") as f:
        bot.send_document(chat_id, f, caption="DanceFlow Choreography Plan")
```

---

### C. Secondary Output — Interactive HTML Dashboard (via `/web`)

Sent only when user types `/web`. A self-contained HTML file with an **animated SVG stage** as the centrepiece.

#### Key Interaction Design

- The **stage SVG** is always visible at the top.
- A **section selector** (row of clickable buttons, one per song section) sits below the stage.
- Clicking a section button **animates** every dancer dot from their current position to that section's target position using the Web Animations API.
- A **motion trail** (dashed orange line) fades out after the animation completes.
- The formation name label below the stage updates live (e.g. `Diagonal → V-shape`).
- A **movement list panel** to the right shows the beat-level cues for the selected section.
- No separate formation cards section — the animated stage replaces all text-based transition descriptions.

#### HTML Template

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DanceFlow — Choreography Dashboard</title>
<style>
  :root {
    --bg: #F9F7F4; --surface: #FFFFFF; --border: #E2DDD7;
    --accent: #D4783A; --text: #2C2C2C; --muted: #8A8A8A;
    --low: #9E9E9E; --medium: #5B9BD5; --high: #E07B39;
    --climax: #C0392B; --rising: #7B68EE;
  }
  * { box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: var(--bg);
         margin: 0; padding: 20px; color: var(--text); }
  h1 { font-size: 20px; margin-bottom: 2px; }
  .meta { font-size: 12px; color: var(--muted); margin-bottom: 24px; }

  /* Layout */
  .main-grid { display: grid; grid-template-columns: 1fr 340px; gap: 20px; }

  /* Stage panel */
  .stage-panel { background: var(--surface); border: 1px solid var(--border);
                 border-radius: 12px; padding: 20px; }
  .stage-title { font-size: 13px; font-weight: 700; margin-bottom: 10px; }
  #stage-svg { width: 100%; display: block; }
  .formation-label { text-align: center; font-size: 13px; color: var(--accent);
                     font-weight: 600; margin-top: 10px; min-height: 20px; }

  /* Section buttons */
  .section-btns { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 16px; }
  .sec-btn {
    padding: 7px 14px; border: none; border-radius: 20px;
    font-size: 12px; font-weight: 600; cursor: pointer;
    color: #fff; transition: transform 0.15s, opacity 0.15s;
  }
  .sec-btn:hover { transform: scale(1.05); }
  .sec-btn.active { outline: 3px solid var(--text); outline-offset: 2px; }

  /* Cue panel */
  .cue-panel { background: var(--surface); border: 1px solid var(--border);
               border-radius: 12px; padding: 20px; overflow-y: auto;
               max-height: 500px; }
  .cue-panel h3 { font-size: 14px; margin: 0 0 14px; }
  .dancer-block { margin-bottom: 16px; }
  .dancer-name { font-size: 12px; font-weight: 700; margin-bottom: 4px;
                 padding: 3px 8px; background: #3D3D3D; color: #fff;
                 border-radius: 4px; display: inline-block; }
  .cue-row { font-size: 11px; padding: 3px 0;
             border-bottom: 1px solid var(--border); }
  .beat-tag { font-weight: 700; color: var(--accent); margin-right: 6px; }

  /* Timeline bar */
  .timeline-wrap { margin-top: 20px; background: var(--surface);
                   border: 1px solid var(--border); border-radius: 12px;
                   padding: 16px; }
  .timeline-bar { display: flex; height: 28px; border-radius: 6px;
                  overflow: hidden; cursor: pointer; }
  .tl-block { display: flex; align-items: center; justify-content: center;
              font-size: 10px; font-weight: 600; color: #fff;
              transition: opacity 0.2s; }
  .tl-block:hover { opacity: 0.8; }
</style>
</head>
<body>

<h1>🎵 DanceFlow — Choreography Dashboard</h1>
<p class="meta" id="meta-line"></p>

<div class="main-grid">

  <!-- Left: Stage + Section Controls -->
  <div>
    <div class="stage-panel">
      <div class="stage-title">Stage Formation</div>
      <svg id="stage-svg" viewBox="0 0 500 300"
           xmlns="http://www.w3.org/2000/svg"></svg>
      <div class="formation-label" id="formation-label">Starting Formation</div>
      <div class="section-btns" id="section-btns"></div>
    </div>

    <!-- Timeline -->
    <div class="timeline-wrap">
      <div class="timeline-bar" id="timeline-bar"></div>
    </div>
  </div>

  <!-- Right: Movement Cues -->
  <div class="cue-panel">
    <h3 id="cue-title">Select a section to see movement cues</h3>
    <div id="cue-content"></div>
  </div>

</div>

<script>
// ── INJECTED DATA ────────────────────────────────────────────────────────────
const SONG_INFO = {
  title: "Song Title", artist: "Artist", bpm: 96, duration_s: 234,
  sections: [
    // { label, start_s, end_s, energy }
  ]
};

const DANCERS = [
  // { id, name, x_norm, y_norm }   ← starting positions
];

// Position of every dancer at every section
// { section_label: [ {id, x_norm, y_norm}, … ] }
const FORMATION_STATES = {};

// Beat-level movement plan
// [ { dancer_id, section, movements: [{beat, time_s, action}] } ]
const MOVEMENT_PLAN = [];
// ─────────────────────────────────────────────────────────────────────────────

const SECTION_COLORS = {
  "Intro": "#9E9E9E", "Verse 1": "#5B9BD5", "Verse 2": "#5B9BD5",
  "Pre-Ch": "#7B68EE", "Chorus": "#E07B39", "Bridge": "#C0392B",
  "Outro": "#9E9E9E"
};
const DEFAULT_COLOR = "#888";

const SVG_W = 500, SVG_H = 300, PAD = 40;

// ── STAGE MAP ────────────────────────────────────────────────────────────────
function toSVG(x_norm, y_norm) {
  return {
    x: PAD + x_norm * (SVG_W - PAD * 2),
    y: PAD + y_norm * (SVG_H - PAD * 2)
  };
}

function buildStage() {
  const svg = document.getElementById("stage-svg");
  svg.innerHTML = `
    <defs>
      <marker id="arr" markerWidth="6" markerHeight="6"
              refX="6" refY="3" orient="auto">
        <path d="M0,0 L6,3 L0,6 Z" fill="#D4783A"/>
      </marker>
    </defs>
    <rect x="${PAD}" y="${PAD}"
          width="${SVG_W-PAD*2}" height="${SVG_H-PAD*2}"
          fill="#F5F2EE" stroke="#C0BAB3" stroke-width="2" rx="4"/>
    <text x="${SVG_W/2}" y="${SVG_H-10}" text-anchor="middle"
          font-size="11" fill="#C0BAB3">STAGE FRONT</text>`;

  // Trail group (drawn under dots)
  svg.innerHTML += `<g id="trail-group"></g>`;

  // Dancer dots
  DANCERS.forEach(d => {
    const p = toSVG(d.x_norm, d.y_norm);
    svg.innerHTML += `
      <circle id="dot-${d.id}" cx="${p.x}" cy="${p.y}" r="18"
              fill="#3D3D3D" stroke="#fff" stroke-width="2"/>
      <text id="lbl-${d.id}" x="${p.x}" y="${p.y+4}"
            text-anchor="middle" font-size="11"
            font-weight="700" fill="#fff" pointer-events="none">
        ${d.name || d.id}
      </text>`;
  });
}

// ── ANIMATE FORMATION ────────────────────────────────────────────────────────
let currentSectionIdx = -1;

function animateToSection(sectionLabel) {
  const targets = FORMATION_STATES[sectionLabel];
  if (!targets) return;

  const trailGroup = document.getElementById("trail-group");
  trailGroup.innerHTML = "";   // clear old trails

  targets.forEach(target => {
    const dot = document.getElementById(`dot-${target.id}`);
    const lbl = document.getElementById(`lbl-${target.id}`);
    if (!dot) return;

    const fromX = parseFloat(dot.getAttribute("cx"));
    const fromY = parseFloat(dot.getAttribute("cy"));
    const tp = toSVG(target.x_norm, target.y_norm);

    // Draw dashed trail line
    const trail = document.createElementNS("http://www.w3.org/2000/svg", "line");
    trail.setAttribute("x1", fromX); trail.setAttribute("y1", fromY);
    trail.setAttribute("x2", tp.x);  trail.setAttribute("y2", tp.y);
    trail.setAttribute("stroke", "#D4783A");
    trail.setAttribute("stroke-width", "1.5");
    trail.setAttribute("stroke-dasharray", "5,4");
    trail.setAttribute("marker-end", "url(#arr)");
    trail.setAttribute("opacity", "0.7");
    trailGroup.appendChild(trail);

    // Animate dot + label using Web Animations API
    const anim = dot.animate(
      [{ cx: `${fromX}`, cy: `${fromY}` },
       { cx: `${tp.x}`,  cy: `${tp.y}`  }],
      { duration: 800, easing: "ease-in-out", fill: "forwards" }
    );
    // Web Animations API animates SVG presentation attrs via KeyframeEffect
    // Fallback: also set attribute at end
    anim.onfinish = () => {
      dot.setAttribute("cx", tp.x);
      dot.setAttribute("cy", tp.y);
      lbl.setAttribute("x", tp.x);
      lbl.setAttribute("y", tp.y + 4);
      // Fade out trails after animation
      trail.animate([{ opacity: "0.7" }, { opacity: "0" }],
                    { duration: 600, delay: 200, fill: "forwards" });
    };

    lbl.animate(
      [{ x: `${fromX}`, y: `${fromY+4}` },
       { x: `${tp.x}`,  y: `${tp.y+4}`  }],
      { duration: 800, easing: "ease-in-out", fill: "forwards" }
    ).onfinish = () => {
      lbl.setAttribute("x", tp.x);
      lbl.setAttribute("y", tp.y + 4);
    };
  });

  // Update formation label
  const prevLabel = currentSectionIdx >= 0
    ? SONG_INFO.sections[currentSectionIdx].label : "Start";
  document.getElementById("formation-label").textContent =
    `${prevLabel} → ${sectionLabel}`;
}

// ── SECTION BUTTONS ──────────────────────────────────────────────────────────
function buildSectionButtons() {
  const container = document.getElementById("section-btns");
  SONG_INFO.sections.forEach((s, i) => {
    const btn = document.createElement("button");
    btn.className = "sec-btn";
    btn.textContent = `${s.label} (${s.start_s}s)`;
    btn.style.background = SECTION_COLORS[s.label] || DEFAULT_COLOR;
    btn.onclick = () => {
      document.querySelectorAll(".sec-btn").forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      animateToSection(s.label);
      currentSectionIdx = i;
      renderCues(s.label);
    };
    container.appendChild(btn);
  });
}

// ── TIMELINE BAR ─────────────────────────────────────────────────────────────
function buildTimeline() {
  const bar = document.getElementById("timeline-bar");
  const total = SONG_INFO.duration_s;
  SONG_INFO.sections.forEach((s, i) => {
    const pct = ((s.end_s - s.start_s) / total * 100).toFixed(1);
    const div = document.createElement("div");
    div.className = "tl-block";
    div.style.width = pct + "%";
    div.style.background = SECTION_COLORS[s.label] || DEFAULT_COLOR;
    div.textContent = pct > 7 ? s.label : "";
    div.title = `${s.label}: ${s.start_s}s–${s.end_s}s`;
    div.onclick = () => {
      document.querySelectorAll(".sec-btn")[i]?.click();
    };
    bar.appendChild(div);
  });
}

// ── MOVEMENT CUES ─────────────────────────────────────────────────────────────
function renderCues(sectionLabel) {
  const section = SONG_INFO.sections.find(s => s.label === sectionLabel);
  document.getElementById("cue-title").textContent =
    `${sectionLabel} — ${section ? section.start_s + "s–" + section.end_s + "s" : ""}`;

  let html = "";
  DANCERS.forEach(d => {
    const plan = MOVEMENT_PLAN.find(
      p => p.dancer_id === d.id && p.section === sectionLabel
    );
    const name = d.name || d.id;
    html += `<div class="dancer-block">
      <span class="dancer-name">${name}</span>`;
    if (plan && plan.movements.length) {
      plan.movements.forEach(m => {
        html += `<div class="cue-row">
          <span class="beat-tag">B${m.beat}</span>${m.action}
        </div>`;
      });
    } else {
      html += `<div class="cue-row" style="color:#ccc">No cues</div>`;
    }
    html += `</div>`;
  });
  document.getElementById("cue-content").innerHTML = html;
}

// ── META LINE ─────────────────────────────────────────────────────────────────
function buildMeta() {
  document.getElementById("meta-line").textContent =
    `${SONG_INFO.title} · ${SONG_INFO.artist} · BPM ${SONG_INFO.bpm} ` +
    `· ${SONG_INFO.duration_s}s · ${DANCERS.length} dancers · DanceFlow v2.0`;
}

// ── INIT ──────────────────────────────────────────────────────────────────────
buildMeta();
buildStage();
buildSectionButtons();
buildTimeline();
</script>
</body>
</html>
```

---

## 8. Skill File Structure (Hermes / OpenClaw)

```yaml
skill_name: DanceFlow
version: 2.0
target_user: Dance instructors, choreographers, student dance groups, performing arts students
real_world_problem: >
  Planning timestamped movements for multiple dancers across a full song is
  time-consuming and hard to communicate visually. DanceFlow automates this
  from a single formation photo and a song, delivering a PDF with formation
  diagrams directly in Telegram.

input_format:
  image:
    type: still photo (JPG, PNG, WebP)
    content: all dancers in starting formation, full-body view
  song:
    options: [MP3 upload, song name as text, YouTube/Spotify/SoundCloud URL]
  optional:
    dance_style: hint for movement vocabulary
    dancer_names: D1=Name, D2=Name, …
    stage_dimensions: e.g. 8m x 6m
    web_output: type /web to request animated HTML instead of PDF

cv_methods:
  - Image preprocessing: CLAHE brightness normalisation, denoising, blur detection
  - Person detection: YOLOv8 pretrained (COCO) — counts and locates all dancers
  - Pose estimation: MediaPipe Pose per dancer crop — posture and facing direction
  - Formation classification: geometric clustering against known formation templates
  - Song structure analysis: librosa BPM and section detection (MP3) or knowledge lookup

workflow:
  1: Validate and preprocess formation image
  2: Detect all dancers (YOLOv8); assign D1–DN with normalised stage coordinates
  3: Pose estimation per dancer; record posture and facing direction
  4: Classify starting formation from dancer coordinates
  5: Analyse song for BPM, section timestamps, and energy levels
  6: Generate beat-level movement plan per dancer per section
  7: Compute formation state (dancer positions) per section
  8a (default): Render formation SVGs → rasterise to PNG → build PDF → send via Telegram
  8b (/web): Build animated HTML dashboard → send HTML file via Telegram

output_format:
  default: PDF (A3 landscape, sent as Telegram document)
    pages:
      - Cover with starting formation diagram
      - Song overview timeline
      - One page per section: movement table + formation diagram
      - One page per formation transition: before/after SVG plots with movement arrows
  on_request (/web): Self-contained interactive HTML with animated SVG stage

limitation_handling:
  blurry_image: warn user with blur score; attempt processing; flag low-confidence detections
  no_person_found: ask user to resend a clearer full-body frontal or elevated photo
  unknown_song: estimate BPM from audio; mark section labels as approximate
  occluded_dancer: label as DX_partial; note in output; suggest re-photo from above
  more_than_10_dancers: cap at 10; notify user; suggest splitting into sub-groups

ethical_boundary:
  - Positions and timing only. No facial identification or biometric data collected.
  - No appearance, body type, or attractiveness judgements of any kind.
  - No image or audio stored beyond the active Telegram session.
  - All movement plans are AI-generated suggestions. Final choreography authority
    rests entirely with the choreographer.
  - Dancers labelled positionally (D1–DN) unless names explicitly provided by user.
```

---

## 9. Telegram Gateway

### Commands

| Command | Action |
|---------|--------|
| `/start` | Welcome message and instructions |
| `/plan` | Begin a new choreography plan |
| `/web` | Re-send last plan as animated HTML dashboard |
| `/help` | List commands and accepted input formats |

### Conversation Flow

```
User:  /plan

Bot:   👋 Send a photo of your dancers in their starting positions,
       then tell me the song (name, link, or upload MP3).
       Optional: dance style · dancer names · stage size.

User:  [photo] Shape of You - Ed Sheeran · hip-hop · Ana Ben Cara Dev Eva

Bot:   🔍 Analysing…
       ✅ 5 dancers detected | Starting: Diagonal line
       🎵 BPM: 96 | 3:54 | 9 sections
       ⚙️ Building choreography plan and PDF…

Bot:   ✅ DanceFlow Plan Ready

       🎵 Shape of You – Ed Sheeran  |  BPM: 96  |  3:54
       👥 5 dancers  |  Diagonal line → V-shape (Chorus) → Circle (Bridge)
       📋 147 cues across 9 sections

       📄 PDF attached — open on any device.
       Type /web for the animated interactive dashboard.

Bot:   [sends choreography_plan.pdf]
```

---

## 10. Limitations and Ethical Awareness

| Limitation | Mitigation |
|------------|-----------|
| Occlusion in formation photo | Detect visible dancers only; flag occluded; suggest elevated photo angle |
| Blurry or dark image | Preprocessing corrects mild issues; severe blur triggers re-upload request |
| Unknown/unreleased song | BPM estimated from audio; sections approximate; user can supply timestamps |
| Pose estimation with baggy clothing | Confidence score shown per dancer; user can override posture |
| Movement suggestions are generic | Framed as a starting plan; choreographer reviews and adjusts |
| PDF rendering environment | WeasyPrint requires system fonts; cairosvg requires Cairo library |
| More than 10 dancers | Capped at 10; user notified; sub-group split recommended |

**Ethical boundaries:** No facial identification. No appearance judgements. No data stored beyond the session. All output is clearly labelled as AI-generated suggestions. Choreographer retains full creative authority.
