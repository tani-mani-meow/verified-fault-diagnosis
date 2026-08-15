// Saved audit results and strict-v1 replay. Source artifacts are in outputs/.
export const STATS = [
  {
    "value": "95.56%",
    "label": "Reported held-out accuracy"
  },
  {
    "value": "85/100",
    "label": "Strict sample-grounded releases"
  },
  {
    "value": "0.9600",
    "label": "Mean structured-claim faithfulness"
  },
  {
    "value": "84",
    "label": "Acoustic features"
  }
]

export const DEMO_CARDS = [
  {
    "id": "bearing",
    "title": "Bearing #179 / RELEASE",
    "color": "#3fb950",
    "llm": "A localized defect on a bearing raceway or rolling element is likely present, as indicated by low energy in the frequency band 3000-4000 Hz and low energy near 375-500 Hz. This defect alters the mechanical rotation profile, exciting high-frequency structural resonances and shifting the energy distribution.",
    "verdict": "RELEASE / passed_verification",
    "claims": [
      {
        "feat": "band_energy_ratio_4",
        "says": "low",
        "actual": "low",
        "ok": true
      },
      {
        "feat": "wpt_energy_4",
        "says": "low",
        "actual": "low",
        "ok": true
      }
    ]
  },
  {
    "id": "liv",
    "title": "LIV #679 / RELEASE",
    "color": "#3fb950",
    "llm": "The inlet/suction reed valve is likely failing to seal during the compression stroke, causing pressurized gas to leak back into the suction line. This backflow has shifted the energy distribution toward lower frequencies, as indicated by the lower spectral centroid and spectral rolloff. The energy near 1750-1875 Hz, 2125-2250 Hz, and 6125-6250 Hz is also lower than expected, but these features are not critical for this diagnosis. The reduced volumetric efficiency is consistent with the predicted LIV fault class.",
    "verdict": "RELEASE / passed_verification",
    "claims": [
      {
        "feat": "spectral_centroid",
        "says": "low",
        "actual": "low",
        "ok": true
      },
      {
        "feat": "spectral_rolloff_85",
        "says": "low",
        "actual": "low",
        "ok": true
      },
      {
        "feat": "wpt_energy_15",
        "says": "low",
        "actual": "low",
        "ok": true
      },
      {
        "feat": "wpt_energy_18",
        "says": "low",
        "actual": "low",
        "ok": true
      },
      {
        "feat": "wpt_energy_50",
        "says": "low",
        "actual": "low",
        "ok": true
      }
    ]
  },
  {
    "id": "lov",
    "title": "LOV #965 / ESCALATE",
    "color": "#f85149",
    "llm": "The energy near 500-625 Hz is lower than expected. This suggests that the discharge/outlet reed valve is not sealing properly, causing compressed gas to leak back into the cylinder during the suction stroke. This results in re-expansion losses and elevated high-frequency band energy on the discharge side.",
    "verdict": "ESCALATE / atypical_action_critical",
    "claims": [
      {
        "feat": "wpt_energy_5",
        "says": "low",
        "actual": "low",
        "ok": true
      }
    ]
  },
  {
    "id": "nrv",
    "title": "LOV #993 / ESCALATE",
    "color": "#f85149",
    "llm": "The energy levels across all frequency bands are within normal ranges. No unusual patterns or anomalies were detected in the provided data.",
    "verdict": "ESCALATE / no_verifiable_claims",
    "claims": []
  }
]

export const GRADIENT_BARS = [
  {
    "label": "Class-only",
    "height": 0.33766666666666667,
    "color": "#f85149",
    "release": "17/100 strict releases",
    "escalate": "83/100 escalations",
    "desc": "Mean structured-claim faithfulness from 100 saved responses. Release counts use strict-v1 offline replay."
  },
  {
    "label": "Values-only",
    "height": 0.63,
    "color": "#d29922",
    "release": "48/100 strict releases",
    "escalate": "52/100 escalations",
    "desc": "Mean structured-claim faithfulness from 100 saved responses. Release counts use strict-v1 offline replay."
  },
  {
    "label": "Sample-grounded",
    "height": 0.96,
    "color": "#3fb950",
    "release": "85/100 strict releases",
    "escalate": "15/100 escalations",
    "desc": "Mean structured-claim faithfulness from 100 saved responses. Release counts use strict-v1 offline replay."
  }
]

export const HEATMAP = {
  "rows": [
    "time",
    "freq",
    "wpt",
    "full"
  ],
  "cols": [
    "XGBoost",
    "SVM",
    "RF",
    "LogReg"
  ],
  "grid": [
    [
      0.512744816550237,
      0.30444924857996153,
      0.5319098593034232,
      0.37895101652311297
    ],
    [
      0.8982228916820223,
      0.41052086881045247,
      0.8726216360133299,
      0.5379744102243308
    ],
    [
      0.9711673457877392,
      0.8453290776168891,
      0.974471705331459,
      0.9983331961421849
    ],
    [
      0.9728367724582245,
      0.40897280536728564,
      0.9788702813644103,
      0.9672204571272982
    ]
  ]
}

export const MECHANISM_CHIPS = [
  {
    "name": "Bearing",
    "kurt": 1.9936726093292239,
    "crest": 6.084237337112427,
    "supported": false,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  },
  {
    "name": "Flywheel",
    "kurt": 1.6647138595581055,
    "crest": 6.58185076713562,
    "supported": false,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  },
  {
    "name": "LIV",
    "kurt": 3.6352739334106445,
    "crest": 6.829837083816528,
    "supported": true,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  },
  {
    "name": "LOV",
    "kurt": 1.9099128246307375,
    "crest": 6.11942195892334,
    "supported": false,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  },
  {
    "name": "NRV",
    "kurt": 3.913299560546875,
    "crest": 7.542214632034302,
    "supported": true,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  },
  {
    "name": "Piston",
    "kurt": 1.6828489303588867,
    "crest": 6.22050666809082,
    "supported": false,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  },
  {
    "name": "Riderbelt",
    "kurt": 3.460775375366211,
    "crest": 7.717537879943848,
    "supported": true,
    "desc": "Class-median lexical heuristic. It does not establish causal validity for a recording."
  }
]

export const MECHANISM_THRESHOLDS = {
  "kurtosis": 2.835590887069702,
  "crest_factor": 7.0839262962341305
}

export const PIPELINE_NODES = [
  {
    "id": "classify",
    "name": "Classify",
    "sub": "Acoustic features",
    "desc": "XGBoost predicts one of eight states from 84 features. Recording-session independence remains unverified.",
    "color": "#58a6ff"
  },
  {
    "id": "ground",
    "name": "Ground",
    "sub": "Knowledge base",
    "desc": "Retrieve per-class grounding features and measured sample directions.",
    "color": "#bc8cff"
  },
  {
    "id": "explain",
    "name": "Explain",
    "sub": "Structured response",
    "desc": "The LLM supplies feature-direction claims and candidate prose.",
    "color": "#d29922"
  },
  {
    "id": "verify",
    "name": "Verify",
    "sub": "strict-v1",
    "desc": "Require nonempty claims, every claim passing, no atypical action-critical claim, and no mechanism warning.",
    "color": "#3fb950"
  }
]

export const NOVELTY = [
  {
    "title": "Sample verification",
    "past": "Class priors supply expected directions.",
    "ours": "Compare claims to measured recording features."
  },
  {
    "title": "Release gate",
    "past": "Saved experiments used the historical gate.",
    "ours": "strict-v1 requires every structured claim to pass."
  },
  {
    "title": "Offline evidence",
    "past": "Saved responses preserve per-claim records.",
    "ours": "Replay the same records and inspect changed decisions."
  }
]

export const LIMITS = [
  {
    "title": "Structured claims only",
    "desc": "Diagnostic correctness, causal prose, advice, and model-assigned criticality are outside the guarantee."
  },
  {
    "title": "Dataset scope",
    "desc": "Recording-session independence and performance on new machines remain unverified."
  },
  {
    "title": "Saved experiment",
    "desc": "Replay uses 100 saved responses per condition. Exact prompts and full response envelopes were not retained."
  },
  {
    "title": "Mechanism heuristic",
    "desc": "Class-level word checks provide a conservative guard. Causal validity requires further evidence."
  }
]
