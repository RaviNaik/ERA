/* Seed data for the ERA V5 capstone task board.
 *
 * This file is the source of truth that lives in git. The board keeps your edits in the
 * browser (localStorage). Use "Export" in the board to download an updated copy of this
 * file, then replace it here and commit it.
 *
 * Dates are YYYY-MM-DD. Every due time is 23:59 IST unless `dueTime` says otherwise.
 * Only public handles and public PR numbers go here: no emails, keys or private messages.
 */
window.TASKBOARD_SEED = {
  rev: 1,
  updated: "2026-11-01",
  me: "Ravi Naik",
  repos: {
    lab: "theschoolofai/era-v5-lab",
    fork: "saitej123/era-v5-lab",
    mine: "RaviNaik/era-v5-lab",
    era: "RaviNaik/ERA"
  },
  tasks: [
    /* ---------- Individual: code-go diagnosis ---------- */
    {
      id: "T-001", type: "task", workstream: "Individual: code-go",
      title: "Read TSAI's Gate 1 DM and list every item to fix on PR #36",
      desc: "TSAI sent each student a DM with the Gate 1 result. A partial result has a numbered list of items. The DM is private, so copy the list here.",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-02", week: 2,
      labels: ["gate-1"], refs: [{ kind: "pr", repo: "lab", ref: "36" }],
      comments: [{ at: "2026-11-01T18:00", by: "Ravi Naik", text: "CAP post of 1 Nov: partial results must be fixed in the same PR by Wed 4 Nov 23:59 IST." }]
    },
    {
      id: "T-002", type: "task", workstream: "Individual: code-go",
      title: "Fix each numbered item from the DM in PR #36",
      desc: "Rows below target end your place in the lab. A missing harm result keeps the rule out of spec v1, but you stay in.",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["gate-1"], blockedBy: "T-001", refs: [{ kind: "pr", repo: "lab", ref: "36" }], comments: []
    },
    {
      id: "T-003", type: "task", workstream: "Individual: code-go",
      title: "Run the harm test on the real Go gold set",
      desc: "Gold file code-go_gold.csv (163 documents, interim, below the 200 target) is on Data A's PR #140. Rule: a medicine passes if it changes or drops at most max(1 document, 1%) of the gold set. Read harm_test.py before running it; run with python -I in a scratch folder.",
      status: "blocked", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["harm-test", "gold-set"], blockedBy: "T-060",
      refs: [{ kind: "pr", repo: "lab", ref: "140" }, { kind: "pr", repo: "lab", ref: "36" }],
      comments: [
        { at: "2026-11-01T12:00", by: "Ravi Naik", text: "Blocked: #140 is approved but not merged, and the gold rebuild is still pending. The auto-mode classifier refused to run the unmerged script." },
        { at: "2026-11-01T12:05", by: "Ravi Naik", text: "Fallback if it stays unavailable: write 'gold set interim, not testable' per rule in medicine.md." }
      ]
    },
    {
      id: "T-004", type: "task", workstream: "Individual: code-go",
      title: "Put the harm-test result for each rule into medicine.md",
      desc: "Gate 1 asks for the result of each medicine on the real gold set, written in medicine.md.",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["harm-test"], blockedBy: "T-003", refs: [{ kind: "pr", repo: "lab", ref: "36" }], comments: []
    },
    {
      id: "T-005", type: "task", workstream: "Individual: code-go",
      title: "Re-review row 500 (doc 34133f5414afff81) and mark it full + clean",
      desc: "Gold builder flagged it as 'partial+clean; not a clean vote'. Read all 409 lines; the file is clean (CRLF, Korean comments).",
      status: "done", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2, completedAt: "2026-11-01",
      labels: ["gold-set"],
      refs: [
        { kind: "commit", repo: "mine", ref: "c61c65b" },
        { kind: "comment", repo: "lab", ref: "36#issuecomment-6558523028" }
      ],
      comments: [{ at: "2026-11-01T19:00", by: "Ravi Naik", text: "Replied to Vijay on PR #36. medicine.md disagreement list corrected from 13 to 12." }]
    },
    {
      id: "T-006", type: "task", workstream: "Individual: code-go",
      title: "Manual line-by-line review of all 500 documents",
      desc: "Corrections: doc 498 fix to clean; docs 165, 174, 329, 445 drop to fix (B4, placeholder in a string constant).",
      status: "done", priority: "high", assignees: ["Ravi Naik"], due: "2026-10-31", week: 1, completedAt: "2026-10-30",
      labels: ["diagnosis"],
      refs: [{ kind: "commit", repo: "mine", ref: "7e726a5" }, { kind: "commit", repo: "mine", ref: "9f3b3f0" }, { kind: "pr", repo: "lab", ref: "36" }],
      comments: []
    },
    {
      id: "T-007", type: "task", workstream: "Individual: code-go",
      title: "Decide whether docs 376 and 472 should carry E2 (data blob)",
      desc: "Optional. E2 means data blobs such as long number arrays. Left as is for now.",
      status: "todo", priority: "low", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["diagnosis"], refs: [{ kind: "pr", repo: "lab", ref: "36" }], comments: []
    },
    {
      id: "T-008", type: "task", workstream: "Individual: code-go",
      title: "Check which 29 code-go gold candidates are disputed",
      desc: "Gold coverage lists 29 disputed candidates for code-go (8 earlier). Find out whether any of my rows are among them and whether a re-review would settle them. Never edit a gold set to make a rule pass.",
      status: "todo", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["gold-set"], refs: [{ kind: "pr", repo: "lab", ref: "140" }], comments: []
    },
    {
      id: "T-009", type: "recurring", recurring: "daily", workstream: "Individual: code-go",
      title: "Check PR #36 and the TSAI DM for new comments",
      desc: "Until Gate 1 fixes are accepted.",
      status: "progress", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["watch"], refs: [{ kind: "pr", repo: "lab", ref: "36" }], comments: []
    },

    /* ---------- G10 Post-training and Agentic ---------- */
    {
      id: "T-020", type: "task", workstream: "G10 Post-training",
      title: "PR #18: resolve #178 conflict, special_tokens_v1 Team 3 IDs, no NFKC",
      desc: "Fork PR into Sai Teja's g10-week1-post-training. Fixes the three items in TSAI's review of lab PR #178. Waiting for Sai Teja to merge.",
      status: "review", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-01", week: 2,
      labels: ["special-tokens"],
      refs: [
        { kind: "pr", repo: "fork", ref: "18" },
        { kind: "commit", repo: "mine", ref: "6415241" },
        { kind: "commit", repo: "mine", ref: "e9eee98" },
        { kind: "pr", repo: "lab", ref: "178" }
      ],
      comments: [{ at: "2026-11-01T17:30", by: "Ravi Naik", text: "35 tests pass. Diff shows 523 files because main was merged in; #178 should show only teams/10-post-training once merged." }]
    },
    {
      id: "T-021", type: "task", workstream: "G10 Post-training",
      title: "Send Sai Teja the PR #18 message",
      desc: "Short version drafted: three fixes, plus two notes (his local special_tokens_v1 may be newer; G3 still has to rename slots 05 and 06).",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-02", week: 2,
      labels: ["comms"], refs: [{ kind: "pr", repo: "fork", ref: "18" }], comments: []
    },
    {
      id: "T-022", type: "task", workstream: "G10 Post-training",
      title: "Lab PR #178: merge fork PR #18, rebase on main, get pr-guard green",
      desc: "TSAI asked for: merge conflict fixed, NFKC paragraph replaced, Team 3 IDs. This is the lead's first week 2 task (due Sun 1 Nov).",
      status: "blocked", priority: "high", assignees: ["Sai Teja"], due: "2026-11-01", week: 2,
      labels: ["gate-2", "special-tokens"], blockedBy: "T-020",
      refs: [{ kind: "pr", repo: "lab", ref: "178" }], comments: []
    },
    {
      id: "T-023", type: "task", workstream: "G10 Post-training",
      title: "Ask G3 to rename g10_slot_05 and g10_slot_06 (subagent_call, subagent_return)",
      desc: "Tokenizer v0.1 (#173) only renamed slots 00 to 04. IDs 130827 and 130828 are still unnamed.",
      status: "todo", priority: "medium", assignees: ["Sai Teja"], due: "2026-11-02", week: 2,
      labels: ["special-tokens", "dependency"], refs: [{ kind: "issue", repo: "lab", ref: "132" }, { kind: "pr", repo: "lab", ref: "173" }], comments: []
    },
    {
      id: "T-024", type: "task", workstream: "G10 Post-training",
      title: "Member A: render.py and tests read the IDs from the Team 3 tokenizer file",
      desc: "Remove the 0 to 29 / 000 to 255 numbering. render.py still uses its own IDs in tokenize_markers().",
      status: "todo", priority: "high", assignees: [], due: "2026-11-02", week: 2,
      labels: ["render.py", "member-A"], refs: [], comments: []
    },
    {
      id: "T-025", type: "task", workstream: "G10 Post-training",
      title: "Ask Sai Teja which letter (A to G) each member holds",
      desc: "Rammohan and Srinivasan asked the same; no answer yet. Decide whether to offer Member A (render.py) or the Go SWE environments (D to F).",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-02", week: 2,
      labels: ["coordination"], refs: [], comments: []
    },
    {
      id: "T-026", type: "task", workstream: "G10 Post-training",
      title: "Licence table: every SFT and RL set with source, licence, generator model, decision",
      desc: "Our 40 candidate rows are in: 11 excluded, 2 allowed (CC-BY-SA), the rest TSAI reserves. TSAI confirmed (31 Oct) that extra sets stay as candidates only; none enters a mix without diagnosis, medicine, harm test and a Gate 2 decision.",
      status: "review", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-03", week: 2,
      labels: ["licence"], refs: [{ kind: "pr", repo: "fork", ref: "10" }, { kind: "commit", repo: "mine", ref: "ed666fa" }], comments: []
    },
    {
      id: "T-027", type: "task", workstream: "G10 Post-training",
      title: "Get TSAI confirmation on our own exclusions: NuminaMath-CoT, rStar-Coder, Big-Math",
      desc: "TSAI already confirmed Magicoder and indic-align. These three were excluded on our own reading of the generator models (GPT-4o, Llama, gpt-4o-mini).",
      status: "todo", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-03", week: 2,
      labels: ["licence"], refs: [], comments: []
    },
    {
      id: "T-028", type: "task", workstream: "G10 Post-training",
      title: "Member B: sft/convert.py and sft/test_convert.py (chat JSONL, loss masks, manifest)",
      desc: "Manifest fields: set, licence, generator model, rows, content hash. Test on 1,000 rows.",
      status: "todo", priority: "high", assignees: [], due: "2026-11-03", week: 2,
      labels: ["sft", "member-B"], refs: [], comments: []
    },
    {
      id: "T-029", type: "task", workstream: "G10 Post-training",
      title: "Member C: decontam_check.py with fingerprints v2 and sft/decontam_report.md",
      desc: "Needs Team 9's fingerprints v2 (due Wed 4 Nov). Include the set_sha256. TSAI packs by Fri 13 Nov.",
      status: "todo", priority: "medium", assignees: [], due: "2026-11-04", week: 2,
      labels: ["decontamination", "member-C"], blockedBy: "", refs: [], comments: []
    },
    {
      id: "T-030", type: "task", workstream: "G10 Post-training",
      title: "sft/sft_mix_proposal.md, first version",
      desc: "Code, agentic, tool use, Indic, general: a share and a reason for each. Lead and Member B. Final proposal due Thu 12 Nov.",
      status: "todo", priority: "medium", assignees: ["Sai Teja"], due: "2026-11-04", week: 2,
      labels: ["sft", "gate-2"], refs: [], comments: []
    },
    {
      id: "T-031", type: "task", workstream: "G10 Post-training",
      title: "sft/sft_mix_proposal.md, final proposal",
      desc: "TSAI decides the SFT mix at Gate 2 (Sat 14 Nov).",
      status: "todo", priority: "high", assignees: ["Sai Teja"], due: "2026-11-12", week: 2,
      labels: ["sft", "gate-2"], blockedBy: "T-030", refs: [], comments: []
    },
    {
      id: "T-032", type: "task", workstream: "G10 Post-training",
      title: "Members D, E, F: 20 SWE environments in swe-envs/<env_id>/ + results/swe_gold_check.md",
      desc: "Each: fixed commit, failing test, gold patch. The check is: the test fails before the gold patch and passes after it. Docker needs a Linux machine.",
      status: "todo", priority: "high", assignees: [], due: "2026-11-13", week: 2,
      labels: ["swe-env", "gate-2"], refs: [], comments: []
    },
    {
      id: "T-033", type: "task", workstream: "G10 Post-training",
      title: "Turn swe-task-go into gold-checked Go environment(s)",
      desc: "Proposed contribution to the 20 environments. Existing task: Sum off-by-one in stats.go, run in golang:1.23-alpine with --network none. Needs a fixed upstream commit per environment.",
      status: "todo", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-13", week: 2,
      labels: ["swe-env", "go"], blockedBy: "T-025", refs: [{ kind: "pr", repo: "fork", ref: "12" }, { kind: "commit", repo: "mine", ref: "2addc53" }], comments: []
    },
    {
      id: "T-034", type: "task", workstream: "G10 Post-training",
      title: "Member G: 10 code tasks in rl/tasks/, rl/verifier.py, rl/test_verifier.py",
      desc: "Test one correct and one wrong answer per task. Offer left open to prototype Go RL tasks.",
      status: "todo", priority: "medium", assignees: [], due: "2026-11-13", week: 2,
      labels: ["rl", "member-G"], refs: [], comments: []
    },
    {
      id: "T-035", type: "task", workstream: "G10 Post-training",
      title: "Member A: give the chat template to Team 12 (G12)",
      desc: "Hand-off to the quantization team.",
      status: "todo", priority: "medium", assignees: [], due: "2026-11-12", week: 2,
      labels: ["handoff", "member-A"], refs: [], comments: []
    },
    {
      id: "T-036", type: "task", workstream: "G10 Post-training",
      title: "Member A: run render.py tests on the frozen tokenizer, write results/render_frozen_tokenizer.txt",
      desc: "Team 3 gives the frozen tokenizer on Sat 14 Nov.",
      status: "todo", priority: "medium", assignees: [], due: "2026-11-14", week: 2,
      labels: ["render.py", "gate-2", "member-A"], refs: [], comments: []
    },
    {
      id: "T-037", type: "task", workstream: "G10 Post-training",
      title: "Give licence facts to Team 11 (Release)",
      desc: "Due Wed 4 Nov.",
      status: "todo", priority: "low", assignees: ["Sai Teja"], due: "2026-11-04", week: 2,
      labels: ["handoff"], refs: [], comments: []
    },
    {
      id: "T-038", type: "recurring", recurring: "daily", workstream: "G10 Post-training",
      title: "Daily note in the G10 group (done, blocked, next)",
      desc: "Each work evening: Sun 1 to Wed 4 Nov, Thu 12 and Fri 13 Nov. Include paths and commit ids.",
      status: "progress", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["daily-note"], refs: [], comments: []
    },
    {
      id: "T-039", type: "task", workstream: "G10 Post-training",
      title: "Go chat-format section and Go SWE task (fork PR #12)",
      desc: "Section 'Go application: code-go' in chat_format_v0.md, swe-task-go with Docker runner, 5 Docker-free tests.",
      status: "done", priority: "medium", assignees: ["Ravi Naik"], due: "2026-10-31", week: 1, completedAt: "2026-10-31",
      labels: ["go", "chat-format"], refs: [{ kind: "pr", repo: "fork", ref: "12" }, { kind: "commit", repo: "mine", ref: "2addc53" }], comments: []
    },
    {
      id: "T-040", type: "task", workstream: "G10 Post-training",
      title: "Review Sai Teja's final branch changes (asked of Amit and Ravi)",
      desc: "Done through PR #18.",
      status: "done", priority: "medium", assignees: ["Ravi Naik", "Amit T"], due: "2026-11-01", week: 2, completedAt: "2026-11-01",
      labels: ["review"], refs: [{ kind: "pr", repo: "fork", ref: "18" }], comments: []
    },

    /* ---------- G12 Quantization (week 2 rotation) ---------- */
    {
      id: "T-050", type: "task", workstream: "G12 Quantization (rotation)",
      title: "Join the pinned Telegram group for G12 connects",
      desc: "Axiom chat gives no notifications; connects are announced in the pinned Telegram group.",
      status: "todo", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-02", week: 2,
      labels: ["coordination"], refs: [], comments: []
    },
    {
      id: "T-051", type: "task", workstream: "G12 Quantization (rotation)",
      title: "Read tools.md and scripts/quantize_eval.py in teams/12-quantization",
      desc: "Week 1 result: OLMoE-1B-7B in 7 formats. Q4_K_M shows no measurable loss; Q3_K_M loses 4.2 MMLU points; Q2_K breaks. The eval-mini notebook runs on Kaggle.",
      status: "todo", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-02", week: 2,
      labels: ["g12"], refs: [], comments: []
    },
    {
      id: "T-052", type: "task", workstream: "G12 Quantization (rotation)",
      title: "Choose one small open model and a 4-bit method; agree it with the G12 lead",
      desc: "Rotation task: quantize one small open model to 4-bit. Compare only with a bf16 score from the same runtime and eval set. Keep the router in F32 or BF16.",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-03", week: 2,
      labels: ["g12"], refs: [], comments: []
    },
    {
      id: "T-053", type: "task", workstream: "G12 Quantization (rotation)",
      title: "Quantize to 4-bit and measure the drop on the eval-mini subset",
      desc: "Use Colab, Kaggle or your own machine. For a bigger run, send a job request (JOB_REQUEST.md). Work days: 1 to 4 Nov and 12 to 13 Nov.",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-13", week: 2,
      labels: ["g12"], blockedBy: "T-052", refs: [], comments: []
    },
    {
      id: "T-054", type: "task", workstream: "G12 Quantization (rotation)",
      title: "Add a row to runs/runs.csv by pull request",
      desc: "Each row needs the model revision, tool version, machine and eval set. Due Sat 14 Nov.",
      status: "todo", priority: "high", assignees: ["Ravi Naik"], due: "2026-11-14", week: 2,
      labels: ["g12", "gate-2"], blockedBy: "T-053", refs: [], comments: []
    },
    {
      id: "T-055", type: "recurring", recurring: "daily", workstream: "G12 Quantization (rotation)",
      title: "Daily note in the G12 group (done, blocked, next)",
      desc: "Each work evening, same days as G10.",
      status: "progress", priority: "medium", assignees: ["Ravi Naik"], due: "2026-11-04", week: 2,
      labels: ["daily-note"], refs: [], comments: []
    },

    /* ---------- Dependencies owned by others ---------- */
    {
      id: "T-060", type: "task", workstream: "Dependencies (other teams)",
      title: "Data A: merge lab PR #140 (gold sets, harm_test.py, medicine tools)",
      desc: "TSAI approved on 1 Nov. Waiting for the gold rebuild and the result of pytest teams/01-data-a/tests -q. After merge the gold path becomes teams/01-data-a/gold/ on main.",
      status: "review", priority: "high", assignees: ["Vijay Patil"], due: "2026-11-02", week: 2,
      labels: ["dependency", "gold-set"], refs: [{ kind: "pr", repo: "lab", ref: "140" }], comments: []
    },
    {
      id: "T-061", type: "task", workstream: "Dependencies (other teams)",
      title: "Data A: rebuild all gold files on #140 and post the pytest result",
      desc: "Taken by Tejaskumar Reddy J on 1 Nov.",
      status: "progress", priority: "high", assignees: ["Tejaskumar Reddy J"], due: "2026-11-02", week: 2,
      labels: ["dependency", "gold-set"], refs: [{ kind: "pr", repo: "lab", ref: "140" }], comments: []
    },
    {
      id: "T-062", type: "task", workstream: "Dependencies (other teams)",
      title: "Data A: announce the new gold path on main in CAP",
      desc: "TSAI asked Vijay to announce it after the merge.",
      status: "todo", priority: "medium", assignees: ["Vijay Patil"], due: "2026-11-02", week: 2,
      labels: ["dependency"], blockedBy: "T-060", refs: [], comments: []
    },
    {
      id: "T-063", type: "task", workstream: "Dependencies (other teams)",
      title: "Team 9: fingerprints v2 and the spec-v1 normalize function",
      desc: "Needed for our decontamination report (T-029). Due Wed 4 Nov.",
      status: "todo", priority: "medium", assignees: [], due: "2026-11-04", week: 2,
      labels: ["dependency"], refs: [], comments: []
    },
    {
      id: "T-064", type: "task", workstream: "Dependencies (other teams)",
      title: "Team 3: frozen tokenizer",
      desc: "After the freeze token IDs stay fixed. Needed for the render.py frozen-tokenizer test (T-036).",
      status: "todo", priority: "medium", assignees: ["Rajiv Seelam"], due: "2026-11-14", week: 2,
      labels: ["dependency", "gate-2"], refs: [{ kind: "issue", repo: "lab", ref: "132" }], comments: []
    },

    /* ---------- Lab milestones ---------- */
    {
      id: "M-01", type: "milestone", workstream: "Lab milestones",
      title: "Gate 1 closed",
      desc: "82 students submitted a diagnosis PR: 25 complete, 57 partial. TSAI sent each student a DM.",
      status: "done", priority: "medium", assignees: [], due: "2026-10-31", week: 1, completedAt: "2026-10-31",
      labels: ["gate-1"], refs: [], comments: []
    },
    {
      id: "M-02", type: "milestone", workstream: "Lab milestones",
      title: "Gate 1 fixes due: partial PRs, missing rows, late-joiner deadline",
      desc: "23:59 IST. A PR with fewer rows than the target ends the place in the lab. A missing or stand-in harm result keeps the rule out of spec v1.",
      status: "todo", priority: "high", assignees: [], due: "2026-11-04", week: 2,
      labels: ["gate-1"], refs: [], comments: []
    },
    {
      id: "M-03", type: "milestone", workstream: "Lab milestones",
      title: "Break: no student work (5 to 11 Nov)",
      desc: "TSAI runs queued job requests. Send any job request before Wed 4 Nov.",
      status: "todo", priority: "low", assignees: [], due: "2026-11-05", week: 2,
      labels: ["calendar"], refs: [], comments: []
    },
    {
      id: "M-04", type: "milestone", workstream: "Lab milestones",
      title: "Gate 2 proposals with evidence due",
      desc: "Each team gives its proposal by Thu 12 Nov. Work days: Thu 12 and Fri 13 Nov.",
      status: "todo", priority: "high", assignees: [], due: "2026-11-12", week: 2,
      labels: ["gate-2"], refs: [], comments: []
    },
    {
      id: "M-05", type: "milestone", workstream: "Lab milestones",
      title: "Gate 2 (TSAI decides)",
      desc: "Tokenizer freeze, KE2, config v1, Indic share, dedup design, spec v1.1, SFT mix, quantization bit widths. G10 evidence: render.py passes on the frozen tokenizer; 20 SWE environments pass the gold check; each set has a licence decision, generator model and decontamination report.",
      status: "todo", priority: "high", assignees: [], due: "2026-11-14", week: 2,
      labels: ["gate-2"], refs: [], comments: []
    },
    {
      id: "M-06", type: "milestone", workstream: "Lab milestones",
      title: "Week 3 rotation starts (15 to 21 Nov): G5 for Ravi",
      desc: "Week 4 is core team only. Briefs arrive in the team groups.",
      status: "todo", priority: "low", assignees: ["Ravi Naik"], due: "2026-11-15", week: 3,
      labels: ["calendar"], refs: [], comments: []
    }
  ]
};
