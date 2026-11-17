"""Text normalisation + domain keyword-group features for the Copilot intent model.

Shared by training (ml/copilot/train_intent.py) and the backend so both see identical features.
Keyword groups are a hand-built gazetteer of cue words per intent family; matching groups are appended
as special tokens (e.g. `kwcompare`), which lets the linear model generalise to unseen phrasings.
"""

from __future__ import annotations

import re

SYNONYMS: list[tuple[str, str]] = [
    (r"\bx[\s-]?rays?\b|\bcxr\b|\bradiograph\b|\bfilm\b", " xray "),
    (r"\bmri\b|\bmr\b", " mri "),
    (r"\bgrad[\s-]?cam\s*(\+\+|plus plus)?\b|\bheat[\s-]?maps?\b|\bsaliency\b|\bcam\b", " heatmap "),
    (r"\bai\b|\bensemble\b|\balgorithm\b|\bsystem\b|\bnetwork\b", " model "),
    (r"\btumou?rs?\b", " tumor "),
    (r"\bc\.?t\.?r\b|\bcardio[\s-]?thoracic ratio\b|\bheart to chest ratio\b", " ctr "),
    (r"\bwhat's\b", "what is"),
    (r"\bwhere's\b", "where is"),
    (r"\bit's\b", "it is"),
    (r"\bi'm\b", "i am"),
]

KEYWORD_GROUPS: dict[str, list[str]] = {
    "quality*": ["quality*", "blurry", "valid*", "modality", "ood", "out of distribution", "banner",
                "warning", "router", "routed", "photo", "dark", "exposure", "contrast",
                "positioning", "rotated", "usable", "adequate", "supported", "recognized",
                "technique", "doubtful", "in distribution", "appropriate", "right type", "proper",
                "frontal view", "read this image", "which pipeline", "is this a"],
    "modelinfo": ["trained", "training", "dataset", "data were", "architecture", "backbone",
                  "weights", "version", "onnx", "pretrained", "imagenet", "densenet", "under the hood",
                  "how does the", "how do you work", "model card", "metrics", "resolution",
                  "from scratch", "llm", "chatgpt", "external api", "engine", "pipeline", "loaded",
                  "who trained", "built", "medivision", "starter or full", "which model"],
    "limits": ["limitation*", "reliable", "trust", "caveat", "accura*", "weakness", "validated",
               "rely", "go wrong", "mistake", "fda", "bias*", "risk", "disclaimer", "medical advice",
               "error rate", "failure", "blind spot", "clinical use", "clinically*", "trustworthy",
               "fair*", "performance", "populations", "children", "replace a radiologist",
               "how good", "how well", "how often", "not good at", "careful", "safe to use"],
    "compare": ["compare", "comparison", "previous", "prior*", "before", "last time", "changed",
                "change", "progress*", "improv*", "worse", "better", "stable", "resolved",
                "baseline", "trend*", "interval", "earlier", "longitudinal", "still there",
                "cleared", "versus", "old scan", "last one", "since", "getting", "new or"],
    "normal": ["normal", "abnormal*", "wrong", "okay", " ok", "healthy*", "worried", "concerning",
               "unremarkable", "negative study", "fine", "clean", "pathology", "disease", "intact",
               "within normal limits", "anything at all", "problems"],
    "summary": ["summary", "summar*", "bottom line", "gist", "tl;dr", "takeaway", "conclu*",
                "overview", "headline", "recap", "overall", "big picture", "in short",
                "key points", "impression", "sum it up", "briefly", "one line"],
    "list*": ["list*", "findings", "detected", "detect*", "flagged", "labels", "predictions",
             "predicted", "results", "top", "positive", "present", "outputs", "probabilities",
             "enumerate", "what did you find", "what did you see", "what did the model"],
    "uncert": ["uncertain*", "uncertainty", "disagree*", "agree*", "spread", "variance", "std",
               "robust", "monte carlo", "dropout", "variability", "consistent", "error bar",
               "unsure", "each model", "individual model", "members", "epistemic", "unstable",
               "noisy", "borderline case", "confidence interval", "collectively", "three models", "individual", "outputs of each", "stable prediction",
               "how stable", "out of distribution", "agreement"],
    "conf": ["confident", "confidence", "sure", "probability", "probable", "likely", "likelihood",
             "odds", "certain*", "certainty", "percentage", "score", "chance", "how strong",
             "definite", "borderline", "number for", "trust the"],
    "where": ["where", "location", "located", "side", "left or right", "lobe", "zone*", "region*",
              "upper", "lower", "localize", "localise", "point out", "bilateral", "central",
              "peripheral", "hotspot", "which lung", "which bone", "part of the", "area"],
    "urgency": ["urgent", "urgency", "emergency", "triage", "priority", "critical", "immediate",
                "stat read", "life threatening", "serious", "severe", "severity", "wait",
                "escalate", "routine", "time sensitive", "risk level", "how bad", "quickly",
                "same day", "how worried", "call the clinician", "top of the worklist"],
    "next": ["next", "recommend*", "follow up", "follow-up", "management", "manage*", "order",
             "further", "workup", "referral", "refer", "consult", "re-image", "repeat",
             "proceed", "plan", "guideline*", "monitor", "action", "should we do", "what to do",
             "investigation*", "suggest"],
    "heatmap": ["heatmap", "overlay*", "attention*", "activation*", "visuali*", "saliency",
                "where did the model focus", "model looked", "what the model saw", "occlusion",
                "xai", "explanation map", "explainability", "hot spots", "highlight*",
                "region of interest", "toggle", "show the", "show me the"],
    "measure*": ["measure*", "ctr", "heart size", "enlarged", "how big", "how large", "area",
                "size", "symmetr*", "volume*", "width", "numbers", "quantif*", "mm", "square",
                "ratio", "dimension"],
    "rewrite*": ["rewrite*", "re-write", "concise", "shorten*", "shorter", "simplify", "rephrase*",
                "reword*", "condense*", "expand*", "detailed version", "more detail", "style",
                "regenerate", "edit the", "tighten", "plain english", "brief", "switch to",
                "less technical", "easier to read", "section", "version"],
    "patient": ["patient", "family", "simple*", "simply*", "plain*", "layman", "lay ", "non doctor",
                "not a doctor", "jargon", "everyday", "eli5", "like i am five", "mother", "child",
                "worried patient", "gently", "normal words", "non technical", "dumb it down",
                "for me", "medical training"],
    "diff": ["differential*", "ddx", "else", "alternative*", "mimic*", "instead", "other",
             "similar", "confused with", "distinguish*", "benign", "artifact", "artefact",
             "normal variant", "possibilities", "consider*", "besides", "really"],
    "why": ["why", "reason", "evidence", "basis", "justify", "rationale", "what made",
            "led to", "convinced", "supports", "driving", "triggered", "features suggest",
            "signs indicate", "arrive at", "how come", "how did you decide", "points to",
            "makes this look", "drove"],
    "define": ["what is a", "what is an", "define", "definition", "meaning of", "what does the word",
               "term", "in general", "in medicine", "causes of", "common causes", "what causes",
               "typical appearance", "teach me", "what do you know about", "refer to",
               "how does .* usually look"],
    "explain*": ["explain*", "describe*", "elaborate*", "details", "walk me through", "tell me more",
                "talk me through", "break down", "clarify", "interpret*", "significance",
                "implications", "help me understand", "go into detail", "more info",
                "don't understand", "what am i looking at", "story", "understand", "seeing for",
                "mean by", "meaning of the", "result for", "finding for"],
    "check": ["is there", "any sign", "any evidence", "rule out", "exclude", "excluded", "absent",
              "do you see", "do you detect", "is .* present", "is .* seen", "is .* visible",
              "does it show", "positive for", "negative for", "yes or no", "check for",
              "look for", "did you find", "does the patient have", "do they have"],
    "greet": ["hi", "hello", "hey", "help", "what can you", "who are you", "what are you",
              "thanks", "thank you", "good morning", "good afternoon", "menu", "options",
              "capabilities", "features", "example questions", "get started", "introduce",
              "assist", "commands", "what should i ask", "start", "yo"],
}

_COMPILED = {
    g: [re.compile(k.strip()) if ".*" in k else
        re.compile(r"(?<![a-z])" + re.escape(k.strip().rstrip("*")) + ("" if k.endswith("*") else r"(?![a-z])"))
        for k in kws]
    for g, kws in KEYWORD_GROUPS.items()
}


def normalize(text: str) -> str:
    t = " " + text.lower().strip() + " "
    for pat, rep in SYNONYMS:
        t = re.sub(pat, rep, t)
    t = re.sub(r"[^a-z0-9;?' ]+", " ", t)
    return " ".join(t.split())


def featurize(text: str) -> str:
    """Normalised text + keyword-group tokens (`kw<group>`) for every group that fires."""
    t = normalize(text)
    tags = [f"kw{g}" for g, pats in _COMPILED.items() if any(p.search(t) for p in pats)]
    return t + " " + " ".join(tags)
