# Review of this list against what the instructor and G10 expect (29 Oct 2026)

Re-checked every dataset against its full card (not only the Hub licence tag).

## What the instructor and the group asked for
- The licence table is for **SFT and RL datasets**, chosen by the lead and the team, **commercial-use data only**, each with licence, source and a decision (instructor, 26-27 Oct).
- Pull requests go to Sai Teja's branch `g10-week1-post-training`; the diagnosis task stays separate and individual.
- Instructor, 28 Oct: **no new datasets this week**; finish the week-1 items with the datasets already chosen.
- Our own focus choice: coding, Indic languages, math, science.

## Result of the re-check
| Check | Result |
|---|---|
| Only SFT or RL data, no pre-training corpora | Pass. `ai4bharat/sangraha` and `Nemotron-Personas-India` were left out for this reason. |
| Licence, source and decision recorded per dataset | Pass: each `dataset.md` has them. |
| Commercial-use only | **One failure found and fixed:** `derek-thomas/ScienceQA` has a Hub tag of cc-by-sa-4.0, but its card says CC BY-NC-SA 4.0 and "non-commercial research purposes only". Removed from the folder. |
| Generator-model terms | Open question for the team, see below. |
| Matches the "no new datasets this week" instruction | **Does not.** This is a pool for later use or for Sai Teja's call; nothing here has been added to the group files. |

## Corrections made after reading the full cards
- `ScienceQA`: removed (non-commercial per its card).
- `ai4bharat/indic-align`: downgraded to "check (high risk)". The card confirms Llama-2-70B-Chat wrote most responses, and the Llama 2 licence forbids using its outputs to improve other LLMs.
- `AceCode-87K`: the card says GPT-4o-mini rewrote every question, so OpenAI terms apply.
- `NuminaMath-CoT`: I had written that GPT-4o generated the solutions. The card does not say that; corrected.
- `open-r1/codeforces`: card header says cc-by-4.0, card body says ODC-By 4.0 (both permissive); problem text belongs to Codeforces.

## Still open (needs a team or instructor decision)
1. **Proprietary-API outputs** (OpenAI: Magicoder, AceCode, Orca-Math Hindi; possibly CodeFeedback sources). Amit's merged table already includes similar sets, so the group has been following the licence tag.
2. **Share-alike** data (`Nemotron-Math-v2` part, `Nemotron-Math-Proofs-v1`, cc-by-sa-4.0).
3. **Problem text from the web** (NuminaMath: exam PDFs and forums; Codeforces; AoPS): the dataset licence may not cover the underlying text.
4. **Unknown generators:** `Nemotron-Terminal-Corpus`, `samvaad-hi-v1`, `oss-codes/*`, `R2E-Gym`, `aya_collection` (translation tools), and `rStar-Coder` (paper not read).
5. Two sets are gated and their samples could not be fetched: `Nemotron-Post-Training-Dataset-v2`, `Big-Math-RL-Verified`.
6. `gsm8k` and `hendrycks_math` are standard benchmarks: marked eval only, not for training. Decontaminate the other math sets against their test problems. Full card-level audit of all sets: see `BENCHMARK_OVERLAP.md`.
