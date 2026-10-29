# Benchmark overlap check (29 Oct 2026)

Scope: do any datasets in this folder contain, or derive from, the problems of standard evaluation benchmarks (GSM8K, MATH, AIME/AMC, HumanEval, MBPP, LiveCodeBench, CodeElo, SWE-bench, GPQA, Terminal-Bench)?

Method: I read the full Hugging Face card of 40 of the 42 datasets and noted what each says about decontamination, benchmark splits and sources. **No text-overlap test was run.** Two datasets (`Big-Math-RL-Verified`, `Nemotron-Post-Training-Dataset-v2`) are gated and their cards could not be read.

Evidence labels: **card** = stated on the dataset card; **not stated** = card says nothing either way (this is not evidence of no overlap); **recollection** = from my memory of the paper, not verified here.

## Result in one line

Only two datasets are benchmarks themselves (GSM8K, MATH, already marked eval only). Most others either state decontamination or say nothing. A few contain benchmark-derived problems by construction or carry a stated residual overlap. Before any training run, run our own n-gram decontamination against the test sets we will report.

## High: contains or derives from benchmark material

| Dataset | Finding | Evidence | Action |
|---|---|---|---|
| openai/gsm8k, EleutherAI/hendrycks_math | The benchmarks themselves | card | eval only (done) |
| nvidia/OpenMathInstruct-2 | Built from GSM8K and MATH **train** problems plus augmentations. Card links a contamination explorer against GSM8K, MATH, AMC 2023, AIME 2024, Omni-MATH | card | Fine only if we never report GSM8K/MATH train-distribution gains as generalisation; run decontamination against AMC/AIME/Omni-MATH |
| project-numina/NuminaMath-CoT | Sources include `gsm8k` (7,345 rows), `math`, `amc_aime` (4,072), `olympiads`. No decontamination statement | card | Decontaminate before use; AIME/AMC items may match AIME eval sets |
| project-numina/NuminaMath-1.5 | Sources include `amc_aime` (5,872) and `olympiads`; no decontamination statement | card | Same |
| open-r1/OpenR1-Math-220k | Problems come from NuminaMath 1.5, so it inherits the above | card | Same |
| agentica-org/DeepCoder-Preview-Dataset | Training set includes 600 LiveCodeBench v5 problems (1 May 2023 to 31 Jul 2024). Its `test` data is LiveCodeBench v5 (1 Aug 2024 to 1 Feb 2025) and CodeElo Codeforces | card | Use `train` only. Do not report LiveCodeBench results before 1 Aug 2024. Treat the `test` configs as eval |
| nebius/SWE-rebench-V2 | Card warning: original SWE-bench repos excluded, **other benchmarks not filtered** (e.g. SWE-bench Multilingual); links an overlap analysis | card | Exclude task IDs that overlap with our eval set |
| nvidia/Nemotron-Science-v1 | Synthetic questions written to mimic GPQA-style topics (style, not copies) | card | Low; still run a near-duplicate check against GPQA |

## Medium: no statement, or overlap likely by source

| Dataset | Finding | Evidence | Action |
|---|---|---|---|
| BytedTsinghua-SIA/DAPO-Math-17k | Competition-style problems; card has no data description or decontamination note | not stated | Check against AIME 2024/25 and MATH-500 |
| open-thoughts/OpenThoughts3-1.2M | Math, code, science mix; card describes deduplication only | not stated | Check against AIME, MATH-500, LiveCodeBench, GPQA |
| nvidia/Nemotron-Post-Training-Dataset-v1 | Code prompts come from OpenCodeReasoning-2 and others; no decontamination statement | not stated | Check |
| nvidia/Nemotron-Math-Proofs-v1 | AoPS, Math StackExchange, MathOverflow problems; no decontamination statement found | not stated | Check |
| microsoft/rStar-Coder | Seeds from competitive-programming sites; card quotes results on LiveCodeBench, HumanEval, MBPP but no decontamination note | not stated (paper may say more: recollection) | Check against LiveCodeBench and CodeElo |
| TIGER-Lab/AceCode-87K | Questions come from several instruction sets (`source` field), rewritten by an LLM; HumanEval/MBPP-style tests | not stated | Check against HumanEval and MBPP |
| m-a-p/CodeFeedback-Filtered-Instruction | Filtered from Magicoder-OSS-Instruct, a ShareGPT Python subset, Magicoder-Evol-Instruct and Evol-Instruct-Code; no decontamination statement. Evol-Instruct-style code sets have had HumanEval-contamination concerns | not stated (concern: recollection) | Check against HumanEval and MBPP |
| ise-uiuc/Magicoder-OSS-Instruct-75K | Empty card. The paper reports decontamination against HumanEval-style evals | recollection | Verify |
| bigcode/self-oss-instruct-sc2-exec-filter-50k | Empty card. The StarCoder2 work reports decontamination | recollection | Verify |
| bigcode/commitpackft | GitHub commits; the same group released HumanEvalPack. Low chance but not tested | not stated | Low |
| SWE-Gym/SWE-Gym, R2E-Gym/R2E-Gym-Subset | SWE-Gym follows the SWE-bench collection procedure on 11 repos; neither card states repo disjointness from SWE-bench | not stated | Compare `repo` fields with the SWE-bench repos |
| nebius/SWE-rebench-openhands-trajectories, nvidia/Nemotron-SFT-SWE-v3 | Trajectories on SWE-rebench-style tasks; no overlap statement | not stated | Same task-ID check as SWE-rebench-V2 |
| nvidia/Nemotron-Terminal-Corpus | Reports gains on Terminal-Bench 2.0; no overlap statement | not stated | Check task overlap if we report Terminal-Bench |
| BhabhaAI/orca-math-...-hindi-filtered, GenVRadmin/Aryabhatta-Orca-Maths-Hindi | Translations of Orca-Math; seed problems not stated on these cards | not stated | Check against GSM8K/MATH test (after translation, so needs a translated eval or the English source) |
| CohereLabs/aya_collection | Templated versions of many public NLP datasets; may include benchmark-derived tasks. Aya's own eval suite is a separate dataset | not stated | Check which source sets are included before using for an Indic eval |

## Low: decontamination stated on the card, or little reason for overlap

| Dataset | Finding | Evidence |
|---|---|---|
| nvidia/OpenMathReasoning | Benchmark decontamination removed questions | card |
| nvidia/Nemotron-Math-v2 | "Decontaminated to avoid overlap with public benchmarks" | card |
| zwhe99/DeepMath-103K | Semantic-match decontamination against common benchmarks | card |
| open-r1/codeforces-cots | `_decontaminated` configs use 8-gram overlap; the plain configs are not decontaminated | card |
| open-r1/codeforces | Contains a `test` split and Codeforces problems that other benchmarks (LiveCodeBench, CodeElo) also use | card |
| nvidia/OpenCodeReasoning, -2 | "We do not include the test split of CodeContests and open-r1/codeforces" | card |
| nvidia/Nemotron-SFT-Competitive-Programming-v2 | Exercism subset decontaminated against the Aider benchmark | card |
| CohereLabs/aya_dataset, ai4bharat/indic-align, sarvamai/samvaad-hi-v1, oss-codes/* | Human-written or translated chat data; no math or code benchmark source seen. Aya's evaluation set is kept separate | card |

## Not checked
- `SynthLabsAI/Big-Math-RL-Verified` and `nvidia/Nemotron-Post-Training-Dataset-v2` (gated; cards unreadable). Big-Math re-uses problems from existing math sets, so assume overlap until read.

## Rules I suggest for the team
1. Use only the `train` splits. Never use any `test` or `validation` split of a coding or math set for training.
2. For `open-r1/codeforces-cots`, use the `_decontaminated` configs.
3. Run one decontamination pass (13-gram or 8-gram overlap on problem text) against the exact test sets we will report, over everything we pick, and keep the removal counts in the licence table notes.
4. Decide the report set first (for example GSM8K, MATH-500, AIME 2024/25, HumanEval, MBPP, LiveCodeBench after 1 Aug 2024, GPQA-Diamond), since decontamination is only meaningful against known targets.
