# API

Interactive OpenAPI docs: <http://localhost:8000/docs>. Errors are always
`{"error": {"code": "...", "message": "..."}}`. Every response carries
`X-MediVision-Disclaimer`.

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/studies` | Upload PNG/JPG/DICOM (≤ 20 MB, multipart `file`; optional `patient_ref`, `modality_hint`, `indication`, `study_date`, `age`, `sex`, `analyze_now`). DICOM is de-identified. |
| GET | `/api/studies` | List (filters `q`, `level`) |
| GET | `/api/studies/{id}` | Study + full results, de-id report, latest report id, other studies of the patient |
| DELETE | `/api/studies/{id}` | Delete study, findings, reviews, reports, chat, files |
| POST | `/api/studies/{id}/analyze?force=` | Re-run the pipeline (`force` analyses even OOD inputs) |
| GET | `/api/studies/{id}/explain?finding=&method=gradcam\|gradcampp\|occlusion&cmap=` | Overlay URL, peak, box, zone |
| GET | `/api/triage/worklist?level=&q=&modality=` | Urgency-ordered worklist + counts |
| POST | `/api/reports/{study_id}/generate?engine=rules\|experimental&style=` | New report draft |
| GET | `/api/reports/by-study/{study_id}` · `/api/reports/{id}` | Read |
| PUT | `/api/reports/{id}` | Edit sections (`{"sections": {...}}`) |
| POST | `/api/reports/{id}/rewrite` | `{"style": "concise\|detailed\|patient\|standard", "section": "impression"}` |
| POST | `/api/reports/{id}/sign` | `{"signed_by": "..."}` → read-only |
| GET | `/api/reports/{id}/pdf` · `/api/reports/{id}/fhir` | Exports |
| POST | `/api/copilot/{study_id}/chat` | `{"message", "stream": true, "delay_ms"}` → SSE `meta`, `token`…, `done` (answer, intent, actions, suggestions) |
| GET | `/api/copilot/{study_id}/history` | Messages + suggested questions |
| POST | `/api/findings/{id}/review` | `{"action": "accept\|reject\|edit", "new_label", "comment", "reviewer"}` |
| GET | `/api/findings/reviews` · `/api/findings/reviews/export` | Agreement rate · re-training CSV |
| POST | `/api/compare` | `{"prior_id", "current_id", "finding"}` → deltas, CTR change, heatmap diff, summary |
| POST | `/api/enhance` | `{"study_id", "ops": {"simulate_low_dose": {"photons"}, "denoise", "clahe": {"clip"}, "unsharp": {"amount"}, "gamma"}}` |
| POST | `/api/federated/run` | `{"rounds", "local_epochs"}` → SSE `setup`, `round`…, `baseline`, `done` |
| GET | `/api/federated/last` | Last run summary |
| GET | `/api/metrics` | All evaluation JSON (starter + full), dataset licenses, split sizes |
| GET | `/api/model-card` | Triage weights, thresholds, temperatures, headline metrics |
| GET | `/api/dashboard` | KPIs |
| GET | `/api/audit?action=` | Audit log |
| GET | `/api/system/health` | Models loaded (profile, ONNX/PyTorch), versions, local-LLM status |
| POST | `/api/system/reload` | Reload weights from `artifacts/` |
| POST | `/api/demo/load` | Seed demo studies |

Static: `/files/...` (de-identified images, overlays), `/artifacts/...` (plots).
