# SAR-Atlas (`v3.8.0`) User Guide

Welcome to **SAR-Atlas (`v3.8.0`)**, a standalone, pure-Rust small-molecule cheminformatics, 3D conformational design, oral PBPK, and relative binding free-energy (`RBFE`) workbench. This guide complements the bundled PDF manual ([`SAR-Atlas_v3.8.0_User_Manual.pdf`](SAR-Atlas_v3.8.0_User_Manual.pdf)) with quick operational workflows, viewport controls, keyboard shortcuts, and CLI examples.

## 1. Execution Modes

| Mode | Command | Description |
| :--- | :--- | :--- |
| **Desktop Workbench (`egui` + `wgpu`)** | `cargo run --release` (or `-- gui`) | 5-domain (`39`-module) workbench with hardware-instanced chemical space, interactive 2D sketcher, 3D conformer/ROCS studio, and software Painter fallback when no GPU adapter is present |
| **Native ANSI Terminal (`TUI`)** | `cargo run --release -- tui` | Zero-dependency 23-view ANSI terminal interface with one-key `h` HTML dossier export |
| **Headless CLI** | `cargo run --release -- <subcommand> [FLAGS]` | Deterministic CSV/SDF/PDB/HTML/SVG batch pipelines (`pbpk`, `rbfe`, `active-learn`, `pipeline`, `dossier`, `esp`, `fetch-pdb`, `sculpt`, `watermap`, etc.) |
| **Offscreen PNG Screenshot** | `cargo run --release -- gui --screenshot out.png --screenshot-view 1` | Renders view `1` (Chemical Space & SALI), `2` (RBFE Network), or `3` (CYP & Metabolites) headlessly |

## 2. Desktop Controls & Keyboard Shortcuts

### Global Navigation Shortcuts
- **`Ctrl + K`**: Open fuzzy Command Palette across all `39` modules and loaded dataset compounds
- **`Ctrl + B`**: Toggle Left Compound & MRU Selector panel
- **`Ctrl + I`**: Toggle Right Lead Inspector panel
- **`Ctrl + F`**: Toggle SMARTS Substructure Spotlight & Filtering drawer
- **`F1`**: Toggle About SAR-Atlas modal
- **`F11`**: Toggle Zen Full-Canvas mode
- **`Escape`**: Dismiss open Command Palette / About modal first, then exit Zen mode

### 2D Chemical-Space & SALI Viewport
- **Left-Drag**: Pan chemical-space coordinates (`world_min [-100, -100]`, `bounds_span [200, 200]`)
- **Mouse Wheel Scroll**: Zoom continuously (screen-space node radii automatically clamp inside `[3.0, 18.0]` px so points remain legible from cluster zoom-out to single-cliff inspection)
- **Click Node**: Select active lead compound across all 39 synchronized modules
- **Hover Node**: Inspect molecule name, assay target, `pAct`, `QED`, `ESOL LogS`, and `BOILED-Egg` blood-brain/GI prediction
- **SMARTS Spotlight Filter**: Highlights matching substructure nodes with pulsing rings; when idle and no SMARTS spotlight/camera motion is active, the viewport pauses continuous frame repaints to eliminate idle CPU/GPU draw
- **Software Fallback Mode**: Automatically engages an `egui::Painter` 2D grid, SALI cliff edge, and node scatter overlay with an in-canvas status banner whenever `wgpu` initialization is unavailable

### 3D Conformers, ROCS, Macrocyclizer & De Novo Viewports
- **Left-Drag**: 3D trackball orbit (`yaw`, `pitch`)
- **Right-Drag**: Screen-space pan (`conf3d_pan`)
- **Mouse Wheel Scroll**: Zoom (`zoom` px per Ångström)
- **Double-Click Canvas**: Reset orientation and execute **Fit-to-Molecule** (`fit_3d_bounding_zoom`) across the active query and overlay conformers
- **Hover Atom**: Display element symbol, atom index, and calculated **Gasteiger PEOE partial charge** (`e`)
- **`DG Trials` Slider (`2..=20`)**: Debounced on slider release (`drag_stopped()` / `lost_focus()`) so distance-geometry/MMFF ensembles do not recompute mid-drag
- **HUD Elements**: Dynamic bounding-sphere perspective camera (prevents near-plane clipping on elongated PROTACs and macrocycles), bottom-left 3D `X/Y/Z` triad gizmo, and bottom-right dynamic Ångström scale bar (`2 Å` / `5 Å` / `10 Å`)

## 3. Core Interactive Workflows

1. **Chemical Space -> Lead Dossier**: Select a compound in `SALI Cliff Viewport` or `Structure Grid`, inspect BBB (`Kp,uu,brain`) and `Cat/ACAT` oral absorption in `Oral PBPK & Kp,uu`, and export a standalone print-ready report from `Lead Dossier & HTML`.
2. **SMARTS & Scaffold Triage**: Press `Ctrl + F` to highlight functional motifs with pulsing spotlights, or filter by Bemis-Murcko framework, PAINS alerts, Pareto optimality, and Lipinski Rule-of-5 compliance.
3. **Alchemical RBFE & Active Learning**: Switch to `Lomap RBFE & BAR` to inspect biconnected perturbation graphs, cycle closure residuals, and MLE affinity reconciliation, or run `Active-Learn QSAR` for Expected Improvement batch acquisition.
4. **3D Structure-Based & Sculpting Modules**: Fetch public structures in `Public PDB & AFDB Fetch` (`UniProt` / `RCSB`), inspect hydration hotspots in `WaterMap & R-Vectors`, optimize tautomer/flip networks in `H-Bond & Bridge Water`, or interactively pull atoms with `Foldit 3D Sculpt & Puzzles`.

## 4. Essential CLI Flags

- `--input <PATH>` / `-i`: Load `.smi`, `.csv`, `.sdf`, `.sqlite`, or `.saratlas.json` dataset on launch (`gui`, `tui`)
- `--screenshot <PNG_PATH>` & `--screenshot-view <1|2|3>`: Capture headless GPU PNG snapshots
- `--hts-synthetic <N>`: Benchmark synthetic chemical-space dataset size (`gui`)
- `pbpk`: Oral compartmental PBPK curves and `Kp,uu,brain` (`--dose-mg`, `--prandial`, `--regimen`, `--export-csv`)
- `rbfe`: Lomap + multi-lambda Bennett BAR + MLE node reconciliation (`--series`, `--topology`, `--lambda-windows`)
- `active-learn`: Hybrid kernel GP + RF QSAR acquisition (`--kernel`, `--model`, `--acq`, `--batch-size`)
- `pipeline`: 6-stage closed-loop DMTA discovery cascade (`--preset`, `--rounds`, `--target-pdb`, `--offtarget-pdb`, `--ledger-sqlite`)
- `dossier`: Derringer-Suich desirability scorecard and HTML/SVG poster generation (`--profile`, `--top-n`, `--export-html`, `--export-svg`)
