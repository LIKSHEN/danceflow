# DanceFlow — AI Agent Skill for Choreography Planning

An AI agent skill definition that takes a formation photo and a song, then generates a complete timestamped choreography plan with visual formation diagrams — delivered as a print-ready PDF or interactive HTML dashboard.

> **This is not a traditional application.** It is a skill specification document (`SKILL.md`) that instructs an AI agent how to behave as a dance choreography planner. Drop it into any agent that supports skill definitions and it works.

---

## What It Does

A choreographer sends:
1. A **photo** of dancers in their starting formation
2. A **song** (MP3, name, or streaming link)

The agent returns:
- A **PDF** (A3 landscape) with per-section movement tables, formation diagrams, and transition arrows
- An **interactive HTML dashboard** (on request) with animated SVG stage, motion trails, and JSON export
- A **summary message** with dancer count, BPM, section count, and formation type

```
User:  [photo.jpg] + "Shape of You — Ed Sheeran"
Agent: 🔍 5 dancers detected | Starting: Diagonal line
       🎵 BPM: 96 | 3:54 | 9 sections
       📋 147 movement cues generated
       📄 [choreography_plan.pdf]
       Type /web for the animated dashboard.
```

---

## Quick Start (Hermes)

### Step 1: Clone or Download

```bash
git clone https://github.com/<your-username>/choreolens.git
```

### Step 2: Place the Skill File

Hermes expects this exact layout:

```
<your-external-dir>/
  SKILL.md              ← MUST be named exactly "SKILL.md"
```

Or with a category subfolder:

```
<your-external-dir>/
  <category>/
    <skill-name>/
      SKILL.md
```

**Common mistake:** Naming the file anything other than `SKILL.md` (e.g. `my_skill.md`, `choreolens_skill.md`) — Hermes will NOT discover it.

#### Option A: Simple (flat directory)

```
C:\Users\you\.hermes\skills\my-custom-skills\
  SKILL.md
```

#### Option B: With category (recommended)

```
C:\Users\you\.hermes\skills\creative\
  danceflow\
    SKILL.md
```

#### Option C: Anywhere on your filesystem

You can put skills in any directory — even `C:\Users\you\Desktop\danceflow\SKILL.md` — as long as you point `external_dirs` to it (see Step 3).

**Key rules:**
- The file **MUST** be named `SKILL.md` (case-sensitive on Linux, case-insensitive on Windows)
- The file **MUST** be directly inside the directory pointed to by `external_dirs`, or inside a subdirectory of it (Hermes scans recursively)
- The frontmatter `---` must be the very first bytes — no leading blank lines or BOM

### Step 3: Register the Directory

Tell Hermes where to find the skill by adding the directory to `external_dirs` in your Hermes config:

```json
{
  "external_dirs": [
    "C:\\Users\\you\\.hermes\\skills\\creative"
  ]
}
```

### Step 4: Verify

Restart Hermes. The skill should appear in your available skills list. Send a formation photo and a song name to test it.

### Dependencies (Runtime)

The agent will need these Python packages available in its execution environment:

| Package | Purpose |
|---------|---------|
| `opencv-python` | Image preprocessing, blur detection |
| `ultralytics` | YOLOv8 person detection |
| `mediapipe` | Pose estimation |
| `librosa` | Audio BPM and section analysis |
| `cairosvg` | SVG-to-PNG rasterisation (requires Cairo library) |
| `playwright` | HTML-to-PDF rendering (`python -m playwright install chromium`) |

---

## Pipeline

The skill defines a 9-step pipeline. Each step is a precise instruction to the agent — embedded Python code serves as an algorithmic specification, not a library to import.

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER INPUT                               │
│            Formation photo  +  Song (MP3/name/URL)              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────▼────────────┐
              │  1. Validate Image      │  Laplacian blur check,
              │                         │  confirm persons present
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  2. Preprocess Image    │  CLAHE brightness,
              │                         │  denoise, upscale to 640px
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  3. Detect Dancers      │  YOLOv8 person detection
              │                         │  → D1–DN, (x, y) coords
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  4. Pose Estimation     │  MediaPipe Pose → facing
              │                         │  direction + posture
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  5. Classify Formation  │  Geometric matching:
              │                         │  line, V, grid, circle…
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  6. Analyse Song        │  librosa BPM + sections,
              │                         │  or reference file lookup
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  7. Generate Movements  │  Beat-level cues per
              │                         │  dancer per section
              └────────────┬────────────┘
                           │
              ┌────────────▼────────────┐
              │  8. Formation States    │  (x, y) positions for
              │                         │  every section
              └────────────┬────────────┘
                           │
                   ┌───────┴───────┐
                   │               │
          ┌────────▼──────┐  ┌─────▼──────────┐
          │  8a. PDF      │  │  8b. HTML      │
          │  (default)    │  │  (/web cmd)    │
          │  Playwright   │  │  Animated SVG  │
          │  + cairosvg   │  │  Web Anim API  │
          └────────┬──────┘  └─────┬──────────┘
                   │               │
                   └───────┬───────┘
                           │
              ┌────────────▼────────────┐
              │  9. Deliver             │  Telegram Bot API
              └─────────────────────────┘
```

### Key Pipeline Details

| Step | Tool / Library | Why |
|------|---------------|-----|
| Blur detection | OpenCV Laplacian variance | Threshold 100 — cheap, reliable |
| Brightness fix | CLAHE in LAB colour space | Fixes underexposed rehearsal photos without colour shift |
| Person detection | YOLOv8 (COCO pretrained) | Fast, accurate `person` class at confidence > 0.5 |
| Pose estimation | MediaPipe Pose | Lightweight, runs on CPU, extracts facing + posture |
| Formation classification | Geometric R-squared matching | Straight line (R² > 0.95), V-shape, grid, diagonal, circle, scattered |
| Song analysis | librosa energy curves | BPM detection + section boundaries from amplitude envelope |
| PDF rendering | Playwright/Chromium + cairosvg | Reliable layout; **not WeasyPrint** (layout drift issues) |
| HTML dashboard | Self-contained SVG + Web Animations API | Single file, no dependencies, works offline |

---

## Repository Structure

```
choreolens/
├── SKILL.md                          # The complete skill definition (1060 lines)
├── references/
│   ├── song-shape-of-you.md          # Pre-computed data: Ed Sheeran, 96 BPM, 10 sections
│   └── song-despacito.md             # Pre-computed data: Luis Fonsi, 89 BPM, 10 sections
└── README.md
```

`SKILL.md` is the entire project. The Python code blocks inside it are precise algorithmic specifications — the agent interprets and executes them at runtime. There is no `requirements.txt` or `package.json` because the skill is a document, not an application.

---

## Song Reference Files

The `references/` folder contains pre-computed song data that is more accurate than runtime audio analysis. Each file follows a standard structure:

```markdown
# Song Title — Artist (DanceFlow Reference)

**BPM:** 96 | **Duration:** 3:53 (233s) | **Style:** Pop/Dancehall

## Song Structure
| Section   | Start | End  | Duration | Energy | Key Characteristics |
|-----------|-------|------|----------|--------|---------------------|
| Intro     | 0:00  | 0:12 | 12s      | low    | Marimba riff...     |
| Verse 1   | 0:12  | 0:43 | 31s      | medium | Vocal enters...     |
| ...       | ...   | ...  | ...      | ...    | ...                 |

## Formation Progression Used (5 dancers, hip-hop)
1. Intro → Straight line
2. Verse 1 → Wide arc
...

## Movement Vocabulary Applied
- Intro: isolations, finger snaps, weight shifts
- Verses: groove walks, body rolls, arm waves
...
```

**Adding a new song:** Create `references/song-{slug}.md` following the same structure. The agent will find it by filename pattern and use it instead of estimating from audio.

---

## Adapting for Your Agent

This skill is platform-agnostic at its core. The pipeline (image → CV → song analysis → movement plan → render → deliver) is the same regardless of which agent runs it. Here's how to adapt it:

### For Claude Code

Claude Code reads skill files as project context. Place `SKILL.md` in your project root or a `.claude/skills/` directory.

1. **Remove the Telegram section** — Claude Code delivers output in the terminal or as files in the working directory
2. **Replace `skill_view()` calls** — Claude Code loads the full file at session start; no dynamic reload needed
3. **Adjust the delivery step** — instead of Telegram Bot API, write the PDF/HTML to disk and report the path:
   ```python
   # Instead of: bot.send_document(chat_id, open(pdf_path, 'rb'))
   # Use:
   print(f"PDF generated: {pdf_path}")
   ```
4. **Keep everything else** — the CV pipeline, song analysis, movement generation, and rendering instructions are all delivery-agnostic

### For Hermes / OpenClaw

This skill was built for Hermes. It works as-is:
- Drop `SKILL.md` into the skills directory
- Ensure `references/` is accessible relative to the skill
- Configure Telegram bot token in `~/AppData/Local/hermes/.env`

### For Other Agents (General Pattern)

Any agent that supports system prompts or skill files can use this. The adaptation pattern is:

| What to change | What to keep |
|----------------|-------------|
| Frontmatter format (match your agent's schema) | The entire CV pipeline (steps 1–5) |
| Delivery mechanism (Discord, Slack, web UI, CLI) | Song analysis logic (step 6) |
| `skill_view()` reload pattern (if any) | Movement generation algorithm (step 7) |
| File path conventions | Formation state computation (step 8) |
| Environment variable locations | Output templates (PDF structure, HTML dashboard) |

### Minimal Adaptation Checklist

1. **Frontmatter** — update `name` and `description` to match your agent's skill registration format
2. **Delivery** — replace the Telegram gateway section with your platform's file/message delivery
3. **Dependencies** — ensure these are available in the agent's execution environment:
   - `opencv-python` — image preprocessing
   - `ultralytics` — YOLOv8 person detection
   - `mediapipe` — pose estimation
   - `librosa` — audio analysis
   - `cairosvg` — SVG-to-PNG (requires Cairo library)
   - `playwright` — HTML-to-PDF (`python -m playwright install chromium`)
4. **References** — add song files to `references/` for songs your users will request
5. **Ethics** — review and adjust the ethical boundaries section for your deployment context

---

## Outputs

### PDF (Default)

A3 landscape, print-ready. Structure:

| Page | Content |
|------|---------|
| 1 | Cover — song info, dancer count, starting formation diagram |
| 2 | Song overview — colour-coded section timeline |
| 3–N | One page per section — movement table + formation diagram |
| N+1–end | Formation transitions — before/after positions with movement arrows |

### HTML Dashboard (/web)

Self-contained single HTML file:
- Animated SVG stage with dancer dots
- Section selector buttons with Web Animations API transitions
- Motion trails (dashed orange lines that fade)
- Movement cue panel with beat-level actions
- "Export Plan" button downloads full choreography as JSON

---

## Limitations

| Limitation | Mitigation |
|------------|-----------|
| Occluded dancers | Flagged as `DX_partial`; suggest elevated photo angle |
| Blurry / dark images | Preprocessing corrects mild issues; severe cases trigger re-upload |
| Unknown songs | BPM estimated from audio; sections approximate |
| Baggy clothing | Confidence score shown per dancer; user can override |
| > 10 dancers | Capped at 10; sub-group split recommended |
| Movement suggestions | Framed as starting points; choreographer reviews and adjusts |

---

## Ethical Boundaries

These are hardcoded into the skill definition:

- **No facial identification** or biometric data extraction
- **No appearance, body type, or attractiveness judgements**
- **No data stored** beyond the active session
- All output is **clearly labelled as AI-generated suggestions**
- **Final creative authority rests entirely with the choreographer**

---

## Contributing

To add a new song reference, create `references/song-{slug}.md` following the structure in the existing reference files. Pull requests welcome.
