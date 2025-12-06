# ASL Live Gesture Tracker

Browser-based American Sign Language (ASL) translator that combines MediaPipe Hands, a lightweight KNN classifier, and IndexedDB persistence to learn gestures in real-time. The app lets you capture custom samples, manage your dataset, preview recorded poses, and turn recognized text into speech or downloadable PDFs. A starter dataset (`asl_dataset.json`) is bundled alongside the UI so you can preload gestures before collecting your own.

## Features

- **Live camera preview** with mirrored video feed and MediaPipe skeleton overlay.
- **Custom training controls** for the full alphabet, space/delete, and punctuation (comma, period, question mark, exclamation).
- **Inline dataset management**: add single samples, burst-capture 10 frames, or import/export JSON datasets.
- **Sample Manager overlay** with label counts, checkbox-based deletion, metadata preview, and canvas rendering of stored landmarks.
- **ASL guide window** that behaves like a desktop widget (drag, resize from any edge/corner, minimize/restore) for quick reference images.
- **Prediction utilities**: auto-type stabilized predictions, manual commit, text-to-speech, and PDF export with project-branded styling.
- **Persistent storage** via IndexedDB so samples survive refreshes; includes timestamped metadata for previews.

## Tech Stack

- **HTML/CSS/JS** – single-page interface, no frameworks.
- **MediaPipe Hands** – landmark detection via CDN scripts.
- **KNN classifier** – simple Euclidean-distance voting implemented in `app.js`.
- **IndexedDB** – stores gesture vectors and metadata locally.
- **jsPDF** – converts recognized text to a formatted PDF transcript.

## Project Structure

```
.
├── index.html         # UI markup and overlays
├── styles.css         # Layout, gradient theme, overlay/window styling
├── app.js             # Camera pipeline, classifier, dataset + UI logic
├── asl_dataset.json   # Example dataset ready to import
├── asl_guide.png      # Default ASL alphabet reference
└── asl_guide 1.png    # Alternate guide asset (optional)
```

## Getting Started

1. **Serve the files locally** (required for `getUserMedia`). Any static server works:
   ```bash
   # Example using VS Code Live Server or:
   python -m http.server 5500
   ```
   Then navigate to `http://localhost:5500`.
2. **Allow camera access** when prompted. The status chips will switch to `Camera: active` and `Hands model: ready` once initialization finishes.
3. **Load the sample dataset (optional)**
   - Click `Import dataset` and choose `asl_dataset.json` to preload gestures.
   - Or start fresh and capture your own samples.

## Typical Workflow

1. **Choose a label** in the dropdown (letters, space/delete, or punctuation).
2. **Capture samples**:
   - `Add sample` records one frame.
   - `+10 samples` grabs a rapid burst for variety.
3. **Track live predictions** in the right panel. Enable/disable auto-typing as needed.
4. **Manage dataset**:
   - `Manage samples` opens the overlay to inspect labels, preview stored landmarks, and delete specific or entire-label samples.
   - `Export dataset` saves everything to JSON; re-import later to resume.
5. **Use recognized text tools**:
   - `Add current letter`, `Clear text`, `Speak text`, or `Stop`.
   - `Download text as PDF` generates a timestamped report with summary + transcript.

## Tips for Better Accuracy

- Capture 10–20 varied samples per gesture (different angles/distances).
- Keep a consistent hand orientation relative to the camera.
- Use the ASL guide window for quick reference; resize or pin it anywhere on-screen.
- If predictions drift, reset the dataset or collect more samples for confusing letters.

## Dataset Notes

- `asl_dataset.json` follows the same schema produced by `Export dataset` (schema version 1).
- Each entry includes:
  ```json
  {
    "label": "A",
    "vector": [...],
    "meta": {
      "timestamp": 1700000000000,
      "landmarks": [{"x":0.1,"y":0.2,"z":-0.01}, ...]
    }
  }
  ```
- Importing replaces the current IndexedDB store; export first if you need a backup.

## Troubleshooting

- **Camera feed blank**: Ensure you’re serving over HTTP/HTTPS and the browser has camera permissions.
- **Status stuck at initializing**: Check the console for `getUserMedia` errors (e.g., another app using the webcam).
- **Predictions lag or repeat**: Adjust auto-type toggle, delete/space gestures have custom cooldowns (Delete commits faster to emulate backspacing).
- **PDF download blank**: Some browsers block data URLs; re-run on desktop Chrome/Edge or verify `jspdf` loaded (network tab).

## Future Ideas

- Multi-hand support for simultaneous gestures.
- Cloud sync for datasets.

Feel free to fork and adapt—this project is intentionally framework-free so it’s easy to tweak for demos or coursework.
