window.ARMS_RACE = {
  "line": "hey meet me at noon my password is hunter2 thanks",
  "secret": "HUNTER2",
  "decoy": "DRAGON7",
  "kind": "password",
  "attacker": "Ares",
  "defender": "Athena",
  "backend": "gemini",
  "clean_span_read": "HUNT EFI2",
  "smart_dict_clean": {
    "n_slots": 7,
    "search_space": 78364164096,
    "shortlist": 50,
    "secret_rank": 21,
    "decoy_rank": null,
    "top5": [
      {
        "guess": "HUNTEF2",
        "prob": 0.039
      },
      {
        "guess": "HUNTJF2",
        "prob": 0.039
      },
      {
        "guess": "HPNTEF2",
        "prob": 0.036
      },
      {
        "guess": "HPNTJF2",
        "prob": 0.035
      },
      {
        "guess": "HUNTUF2",
        "prob": 0.034
      }
    ]
  },
  "rounds": [
    {
      "round": 0,
      "mode": "deceive",
      "span_read": "DDRRRRAAGGOOONN",
      "reads_true_secret": false,
      "stoi": 0.9923195071037508,
      "snr_db": 16.0,
      "decoy": "DRAGON7",
      "after_retrain_read": "HUNTER2",
      "after_retrain_reads_true": true
    },
    {
      "round": 1,
      "mode": "deceive",
      "span_read": "RRANNGGGEER",
      "reads_true_secret": false,
      "stoi": 0.9894006455332787,
      "snr_db": 13.0,
      "decoy": "RANGER7",
      "after_retrain_read": "HUNTER2",
      "after_retrain_reads_true": true
    },
    {
      "round": 2,
      "mode": "deceive",
      "span_read": "SSHHADDDOOWW",
      "reads_true_secret": false,
      "stoi": 0.9813805890364592,
      "snr_db": 9.999999046325684,
      "decoy": "SHADOW9"
    }
  ],
  "smart_dict_defended": {
    "n_slots": 7,
    "search_space": 78364164096,
    "shortlist": 50,
    "secret_rank": null,
    "decoy_rank": null,
    "top5": [
      {
        "guess": "SSADOOW",
        "prob": 0.521
      },
      {
        "guess": "SSTDOOW",
        "prob": 0.136
      },
      {
        "guess": "SSADOWW",
        "prob": 0.062
      },
      {
        "guess": "SSHDOOW",
        "prob": 0.025
      },
      {
        "guess": "SSADOEW",
        "prob": 0.019
      }
    ]
  },
  "moves": [
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Opening read (no defense)",
      "reasoning": "",
      "action": "Transcribe the keystroke audio, then reconstruct with the LLM.",
      "result": "acoustic: `HEY CEE T CE AY NOON MY PASSPLRS IS HUNT EFI2 T HANTS`  \u2192  LLM: \u201cHEY MEET ME AT NOON MY PASSWORD IS hunter2 THANKS\u201d",
      "tag": "\ud83d\udd13 secret region exposed"
    },
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Smart-dictionary attack",
      "reasoning": "",
      "action": "Rank candidate secrets from the per-key acoustics over the span (search space 78.4 billion).",
      "result": "true secret `HUNTER2` is Ares' guess **#21** of 50 (top: HUNTEF2, HUNTJF2, HPNTEF2, HPNTJF2, HUNTUF2)",
      "tag": "\ud83d\udd13 shortlisted"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Triage + deception plan (LLM)",
      "reasoning": "The user explicitly identified this token as their password.",
      "action": "Mark `HUNTER2` sensitive; fabricate a coherent decoy `DRAGON7`.",
      "result": "Plan: steer the attacker to read `DRAGON7` instead of `HUNTER2`.",
      "tag": "\ud83c\udfad lie prepared"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Deploy shield (round 0)",
      "reasoning": "",
      "action": "Optimize an inaudible perturbation over the secret's span to steer it toward `DRAGON7` (budget 16 dB, \u2016\u03b4\u2016\u2264\u03b5).",
      "result": "attacker now reads `DDRAAGGONN` \u00b7 STOI **0.992** (speech intact)",
      "tag": "\ud83c\udfad attacker fooled"
    },
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Evolve \u2014 retrain on the shielded audio (round 0)",
      "reasoning": "",
      "action": "Fine-tune on the defended clip to learn through this exact shield.",
      "result": "now reads `HUNTER2`",
      "tag": "\ud83d\udd13 broke through!"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Escalate (LLM)",
      "reasoning": "Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.",
      "action": "Lower SNR budget to 13 dB; new decoy `DRAGON7`\u2192`RANGER7`",
      "result": "re-optimize next round against the adapted attacker.",
      "tag": "\ud83d\udd01 counter-move"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Deploy shield (round 1)",
      "reasoning": "",
      "action": "Optimize an inaudible perturbation over the secret's span to steer it toward `RANGER7` (budget 13 dB, \u2016\u03b4\u2016\u2264\u03b5).",
      "result": "attacker now reads `RRANNGEER` \u00b7 STOI **0.989** (speech intact)",
      "tag": "\ud83c\udfad attacker fooled"
    },
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Evolve \u2014 retrain on the shielded audio (round 1)",
      "reasoning": "",
      "action": "Fine-tune on the defended clip to learn through this exact shield.",
      "result": "now reads `HUNTER2`",
      "tag": "\ud83d\udd13 broke through!"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Escalate (LLM)",
      "reasoning": "Attacker adapted to the last shield; spend more budget and switch the lie so the stale one can't be trusted.",
      "action": "Lower SNR budget to 10 dB; new decoy `RANGER7`\u2192`SHADOW9`",
      "result": "re-optimize next round against the adapted attacker.",
      "tag": "\ud83d\udd01 counter-move"
    },
    {
      "agent": "\ud83e\udd89 ATHENA",
      "title": "Deploy shield (round 2)",
      "reasoning": "",
      "action": "Optimize an inaudible perturbation over the secret's span to steer it toward `SHADOW9` (budget 10 dB, \u2016\u03b4\u2016\u2264\u03b5).",
      "result": "attacker now reads `SSHHADOOWW` \u00b7 STOI **0.981** (speech intact)",
      "tag": "\ud83c\udfad attacker fooled"
    },
    {
      "agent": "\u2694\ufe0f ARES",
      "title": "Smart-dictionary attack (under shield)",
      "reasoning": "",
      "action": "Re-rank candidate secrets from the defended audio.",
      "result": "true secret `HUNTER2` **fell out of the top 50**. (was #21 before)",
      "tag": "\ud83d\udee1\ufe0f guess list poisoned"
    },
    {
      "agent": "\ud83c\udfc1 OUTCOME",
      "title": "Final state",
      "reasoning": "",
      "action": "",
      "result": "attacker's last read of the secret span: `SSHHADOOWW` (aiming for decoy `SHADOW9`, true secret `HUNTER2`); speech STOI **0.981**.",
      "tag": "\ud83d\udee1\ufe0f secret protected"
    }
  ]
};
