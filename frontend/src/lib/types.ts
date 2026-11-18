export type Level = "CRITICAL" | "URGENT" | "ROUTINE" | "NORMAL";
export type Certainty = "Likely" | "Possible" | "Unlikely";

export interface Finding {
  id?: string;
  label: string;
  key: string;
  probability: number;
  uncertainty: number;
  ensemble_std?: number;
  mc_dropout_var?: number;
  threshold: number;
  positive: boolean;
  certainty: Certainty;
  ratio_to_threshold: number;
  member_probs?: Record<string, number>;
  zone?: string | null;
  heat_peak?: [number, number];
  heat_box?: [number, number, number, number];
  box?: [number, number, number, number];
  detector_score?: number;
  kind: string;
  model?: string;
  grade?: string;
  expected_grade?: number;
  risk_category?: string;
  volume_mm3?: number;
  rationale?: { reasons: { kind: string; text: string }[]; summary: string };
  provenance?: { source: string; model: string | null; model_version: string | null };
  review?: {
    status: string;
    edited_label?: string | null;
    note?: string | null;
    reviewed_by?: string | null;
    reviewed_role?: string | null;
    reviewed_at?: string | null;
    source?: string;
  };
}

export interface Radiomics {
  roi: string;
  shape: Record<string, number>;
  first_order: Record<string, number>;
  texture_glcm?: Record<string, number>;
  note?: string;
}

export interface Autonomy {
  enabled: boolean;
  action?: "pre_cleared_normal" | "critical_alert" | "none";
  text?: string;
  requires_human_confirmation?: boolean;
}

export interface Timeline {
  pipeline: string;
  n_studies: number;
  span_days: number;
  points: {
    study_id: string;
    date: string;
    day: number;
    urgency_score: number | null;
    urgency_level: Level | null;
    probabilities: Record<string, number>;
    positive: string[];
    ctr?: number | null;
    lesion_size?: number | null;
  }[];
  trends: { label: string; first: number; last: number; slope_per_30d: number; trend: string; series: number[] }[];
  urgency_slope_per_30d: number;
  lesion_growth: { first: number; last: number; change_pct: number } | null;
  call: string;
  note: string;
}

export interface Me {
  username: string;
  role: "admin" | "radiologist" | "resident" | "viewer";
  name: string;
  authenticated: boolean;
  auth_enabled: boolean;
  permissions: string[];
  roles: string[];
}

export interface Modality {
  label: string;
  probs: Record<string, number>;
  confidence: number | null;
  pipeline: string | null;
  ood: boolean;
  energy_ood?: boolean;
  warning: string | null;
  forced?: boolean;
}

export interface Measurements {
  available?: boolean;
  ctr?: number | null;
  ctr_flag?: boolean;
  heart_width_px?: number;
  thorax_width_px?: number;
  right_lung_area_px?: number;
  left_lung_area_px?: number;
  right_lung_area_mm2?: number | null;
  left_lung_area_mm2?: number | null;
  symmetry_index?: number | null;
  lines?: { thorax: [number, number][]; heart: [number, number][] };
  quality?: string;
  note?: string;
  tumour_area_px?: number;
  tumour_area_mm2?: number | null;
  tumour_fraction?: number;
  box?: [number, number, number, number];
  centroid?: [number, number];
  consistent_with_classifier?: boolean;
  volume_mm3?: number;
  equivalent_diameter_mm?: number;
  slices_with_nodule?: number;
  spacing_assumed?: boolean;
  voxels?: number;
}

export interface Triage {
  score: number;
  level: Level;
  drivers: { label: string; weight: number; evidence: number; confidence: number; contribution: number }[];
  note: string | null;
}

export interface Detection {
  label: string;
  score: number;
  box: [number, number, number, number];
}

export interface Results {
  pipeline: "chest" | "fracture" | "brain" | "head_ct" | "breast_us" | "fundus" | "ct_volume" | null;
  modality: Modality;
  findings: Finding[];
  detections: Detection[];
  measurements: Measurements;
  overlays: {
    heatmaps: Record<string, Record<string, string>>;
    lung_mask?: string;
    tumour_mask?: string;
    volume?: { mpr: Record<string, string>; mip_frames: string[]; shape: number[] };
  };
  triage: Triage;
  extra: {
    chest?: { members: string[]; profile: string; energy: number; energy_threshold: number; energy_ood: boolean };
    brain?: { class_probs: Record<string, number>; predicted: string; no_tumour_prob: number };
    fundus?: { grade_probs: Record<string, number>; predicted_grade: string };
  };
  model_versions?: { name: string; profile: string; version: string; trained_at: string | null; backend: string }[];
  radiomics?: Radiomics | null;
  autonomy?: Autonomy;
  image: { url: string; width: number; height: number };
  model_profile: "starter" | "full";
  model_profiles: Record<string, string>;
  backend: string;
  timings_ms: Record<string, number>;
  report_preview?: string[];
}

export interface Patient {
  id: string;
  pseudonym: string;
  sex: string | null;
  age: number | null;
}

export interface StudySummary {
  id: string;
  patient: Patient | null;
  filename: string;
  format: string;
  modality: string | null;
  modality_hint: string | null;
  status: string;
  urgency_score: number | null;
  urgency_level: Level | null;
  ood: boolean;
  model_profile: string | null;
  indication: string | null;
  study_date: string | null;
  description: string | null;
  image_url: string | null;
  width: number;
  height: number;
  created_at: string;
  analyzed_at: string | null;
  top_finding: { label: string; probability: number; certainty: Certainty } | null;
  n_positive: number;
  pipeline: string | null;
  warning: string | null;
  autonomy?: Autonomy | null;
  alert?: { active: boolean; acknowledged_by: string | null; acknowledged_at: string | null };
  consent_basis?: string | null;
  uploaded_by?: string | null;
  input_sha256?: string | null;
  patient_id?: string;
}

export interface Study extends StudySummary {
  results: Results;
  deid: {
    format?: string;
    removed_tags?: string[];
    pseudonym?: string;
    ms?: number;
    exif_stripped?: boolean;
    volume_shape?: number[] | null;
  };
  report_id: string | null;
  other_studies: { id: string; created_at: string; study_date: string | null; modality: string | null; status: string }[];
}

export type ReportSections = {
  indication?: string;
  technique?: string;
  findings?: Record<string, string>;
  measurements?: string[];
  impression?: string[];
  recommendations?: string[];
  ai_confidence?: string[];
  disclaimer?: string;
  meta?: Record<string, unknown>;
};

export interface Report {
  id: string;
  study_id: string;
  engine: "rules" | "experimental";
  style: string;
  sections: ReportSections;
  status: "draft" | "signed";
  signed_by: string | null;
  signed_role?: string | null;
  signed_at: string | null;
  drafted_by?: string;
  edited_by?: string | null;
  version: number;
  created_at: string;
  updated_at: string;
}

export interface ChatAction {
  label: string;
  type: "show_heatmap" | "open_compare" | "open_tab" | "navigate" | "rewrite_report" | "toggle_overlay";
  finding?: string;
  method?: string;
  tab?: string;
  href?: string;
  a?: string;
  b?: string;
  style?: string;
  section?: string;
  overlay?: string;
}

export interface ChatAnswer {
  text: string;
  intent: string;
  confidence: number;
  focus: string | null;
  actions: ChatAction[];
  suggestions: string[];
  sources: string[];
}

export interface ChatMessage {
  id?: string;
  role: "user" | "assistant";
  content: string;
  intent?: string | null;
  actions?: ChatAction[];
  suggestions?: string[];
  streaming?: boolean;
}

export interface Health {
  status: string;
  version: string;
  disclaimer: string;
  uptime_s: number;
  models: Record<string, { loaded: boolean; profile?: string; backend?: string; img_size?: number }>;
  n_models_loaded: number;
  active_profile: string | null;
  chest_profile: string | null;
  inference_backend: string;
  versions: Record<string, string>;
  local_llm: { enabled: boolean; configured: boolean; url_is_local: boolean; model: string | null };
  external_ai_apis: string;
  settings: Record<string, unknown>;
}

export interface Explain {
  finding: string;
  method: string;
  cmap: string;
  url: string;
  peak: [number, number];
  box: [number, number, number, number];
  zone: string | null;
  model: string;
  ms?: number;
}
