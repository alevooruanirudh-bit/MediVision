# Demo script (≈6 minutes)

Setup: `make dev`, open <http://localhost:3000>, click **Load demo data** (12 studies are uploaded and
analysed by our models).

1. **Landing (30 s)** — "Every model here was trained by this project on free public data; nothing
   calls an external AI service." Point at the live status line: models loaded, ONNX/CPU, *external AI
   APIs: none*.
2. **Worklist (45 s)** — Studies sorted by urgency; CRITICAL pulses. Filter by URGENT, search a
   pseudonym. Drag `sample_data/chest_effusion.png` onto the uploader — watch upload → analysing → Open.
3. **Study viewer (90 s)** — Probability bars with uncertainty whiskers and threshold ticks; click
   another finding → its Grad-CAM appears; switch to Grad-CAM++ and occlusion; change the colormap and
   opacity. Toggle lung masks and CTR lines (Measurements tab shows the CTR gauge). Shift-drag to
   window/level, `i` to invert, `f` for fullscreen. Accept one finding, reject another.
4. **Copilot (60 s)** — Ask "why effusion?", "where is it?", "how confident are you?", "what should we
   do next?", "explain it to the patient". Click the *Show heatmap* chip. Mention: intent model +
   knowledge base + rule composers, streamed over SSE, never a definitive diagnosis.
5. **Report editor (45 s)** — Switch style to *Patient-friendly*, regenerate the Impression, try the
   *Experimental* engine (our Transformer decoder) and note the consistency check. Sign off, export PDF
   and FHIR.
6. **Compare (30 s)** — Patient with prior/current studies: synced zoom, delta table, heatmap
   difference, interval-change paragraph.
7. **OOD guard (15 s)** — Open the horse photo study: router says *not medical*, nothing is analysed.
8. **Enhancement Lab (20 s)** — Simulated low dose at 40 photons, denoiser on: PSNR/SSIM jump; drag
   the before/after slider.
9. **Federated (30 s)** — Run FedAvg; three hospitals stream per-round AUROC; compare with
   centralised and local-only.
10. **Lab + Model card (30 s)** — Per-label AUROC, ROC, calibration before/after temperature scaling,
    fairness by sex/age/view, honest starter numbers; triage weights and limitations.
