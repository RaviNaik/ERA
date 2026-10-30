
One folder per dataset that passed the licence screen (29 Oct 2026). Each folder has `dataset.md` (licence, stage, fields, splits, caveats, card excerpt) and `sample_20.jsonl` (20 rows to inspect).

Decision key: **use** = permissive tag and nothing against it found; **use, check** = permissive tag but a caveat to settle first (generator-model terms, share-alike, mixed sub-licences); **excluded (TSAI ruling)** = quarantined by TSAI on 30 Oct for generator-model terms or unclear provenance; **allowed (CC-BY-SA, attribution needed)** = CC-BY-SA data made by a permissive model, allowed by TSAI's later update; kept listed with the reason, never used in a training mix; **G9 decontamination first** = source goes to G9 (Evaluation) for a decontamination test before any use; **eval only** = standard benchmark, kept out of training; **check** = origin or licence still to verify. Non-commercial and untagged sets are not here.

Benchmark overlap per dataset: see [BENCHMARK_OVERLAP.md](BENCHMARK_OVERLAP.md).

A Hugging Face licence tag is not legal clearance. Samples are small excerpts kept for inspection under each dataset's own licence; see the source link in each `dataset.md`.

| Category | Folder | Stage | Licence tag | Decision | Sample rows | Total rows |
|---|---|---|---|---|---|---|
| coding | [R2E-Gym__R2E-Gym-Subset](R2E-Gym__R2E-Gym-Subset/dataset.md) | RL (SWE tasks) | apache-2.0 | use | 20 | 4578 |
| coding | [SWE-Gym__SWE-Gym](SWE-Gym__SWE-Gym/dataset.md) | RL (SWE tasks) | mit | use | 20 | 2438 |
| coding | [TIGER-Lab__AceCode-87K](TIGER-Lab__AceCode-87K/dataset.md) | RL | mit | excluded (TSAI ruling) | 20 | 87149 |
| coding | [agentica-org__DeepCoder-Preview-Dataset](agentica-org__DeepCoder-Preview-Dataset/dataset.md) | RL | mit | use; G9 decontamination first | 20 | 24974 |
| coding | [bigcode__commitpackft](bigcode__commitpackft/dataset.md) | SFT | mit | use, check | 20 | unknown |
| coding | [bigcode__self-oss-instruct-sc2-exec-filter-50k](bigcode__self-oss-instruct-sc2-exec-filter-50k/dataset.md) | SFT | odc-by | use | 20 | 50661 |
| coding | [ise-uiuc__Magicoder-OSS-Instruct-75K](ise-uiuc__Magicoder-OSS-Instruct-75K/dataset.md) | SFT | mit | excluded (closed-model generator) | 20 | 75197 |
| coding | [m-a-p__CodeFeedback-Filtered-Instruction](m-a-p__CodeFeedback-Filtered-Instruction/dataset.md) | SFT | apache-2.0 | use, check | 20 | 156526 |
| coding | [microsoft__rStar-Coder](microsoft__rStar-Coder/dataset.md) | SFT + RL | cc-by-4.0 | excluded (closed-model generator) | 20 | 1392267 |
| coding | [nebius__SWE-rebench-V2](nebius__SWE-rebench-V2/dataset.md) | RL (SWE tasks) | cc-by-4.0 | use | 20 | 32079 |
| coding | [nebius__SWE-rebench-openhands-trajectories](nebius__SWE-rebench-openhands-trajectories/dataset.md) | SFT (agent trajectories) | cc-by-4.0 | use | 20 | 67074 |
| coding | [nvidia__Nemotron-SFT-Competitive-Programming-v2](nvidia__Nemotron-SFT-Competitive-Programming-v2/dataset.md) | SFT | cc-by-4.0, odc-by, mit | use, check | 20 | unknown |
| coding | [nvidia__Nemotron-SFT-SWE-v3](nvidia__Nemotron-SFT-SWE-v3/dataset.md) | SFT (agent trajectories) | cc-by-4.0, apache-2.0, mit, bsd-3-clause, bsd-2-clause | use | 20 | 237970 |
| coding | [nvidia__Nemotron-Terminal-Corpus](nvidia__Nemotron-Terminal-Corpus/dataset.md) | SFT (terminal) | cc-by-4.0 | use, check | 20 | 366154 |
| coding | [nvidia__OpenCodeReasoning](nvidia__OpenCodeReasoning/dataset.md) | SFT | cc-by-4.0 | use, check | 20 | 337766 |
| coding | [nvidia__OpenCodeReasoning-2](nvidia__OpenCodeReasoning-2/dataset.md) | SFT | cc-by-4.0 | use, check | 20 | 220000 |
| coding | [open-r1__codeforces](open-r1__codeforces/dataset.md) | RL | cc-by-4.0 | use, check | 20 | 29286 |
| coding | [open-r1__codeforces-cots](open-r1__codeforces-cots/dataset.md) | SFT | cc-by-4.0 | use, check | 20 | 253876 |
| indic | [CohereLabs__aya_collection](CohereLabs__aya_collection/dataset.md) | SFT | apache-2.0 | use, check | 20 | 513755590 |
| indic | [CohereLabs__aya_dataset](CohereLabs__aya_dataset/dataset.md) | SFT | apache-2.0 | use | 13 | 205568 |
| indic | [ai4bharat__indic-align](ai4bharat__indic-align/dataset.md) | SFT | cc-by-4.0 | excluded (closed-model generator) | 20 | 97419884 |
| indic | [sarvamai__samvaad-hi-v1](sarvamai__samvaad-hi-v1/dataset.md) | SFT | apache-2.0 | use, check | 20 | 101476 |
| indic, coding | [oss-codes__Coding-Conversational-Dataset-Indic](oss-codes__Coding-Conversational-Dataset-Indic/dataset.md) | SFT | mit | check | 20 | unknown |
| indic, coding | [oss-codes__Computer-Science-Conversational-Dataset-Indic](oss-codes__Computer-Science-Conversational-Dataset-Indic/dataset.md) | SFT | mit | check | 20 | 61736 |
| indic, math | [BhabhaAI__orca-math-word-problems-200k-hindi-filtered](BhabhaAI__orca-math-word-problems-200k-hindi-filtered/dataset.md) | SFT | mit | excluded (TSAI ruling) | 20 | 188943 |
| indic, math | [GenVRadmin__Aryabhatta-Orca-Maths-Hindi](GenVRadmin__Aryabhatta-Orca-Maths-Hindi/dataset.md) | SFT | mit | excluded (TSAI ruling) | 20 | 200000 |
| math | [BytedTsinghua-SIA__DAPO-Math-17k](BytedTsinghua-SIA__DAPO-Math-17k/dataset.md) | RL | apache-2.0 | use | 20 | 1791700 |
| math | [EleutherAI__hendrycks_math](EleutherAI__hendrycks_math/dataset.md) | eval only (benchmark) | mit | eval only | 20 | 12500 |
| math | [SynthLabsAI__Big-Math-RL-Verified](SynthLabsAI__Big-Math-RL-Verified/dataset.md) | RL | apache-2.0 | excluded (closed-model generator) | 0 | unknown |
| math | [nvidia__Nemotron-Math-Proofs-v1](nvidia__Nemotron-Math-Proofs-v1/dataset.md) | SFT | cc-by-sa-4.0 | allowed (CC-BY-SA, attribution needed) | 20 | 243314 |
| math | [nvidia__Nemotron-Math-v2](nvidia__Nemotron-Math-v2/dataset.md) | SFT | cc-by-4.0, cc-by-sa-4.0 | allowed (CC-BY-SA, attribution needed) | 20 | 7085839 |
| math | [nvidia__OpenMathInstruct-2](nvidia__OpenMathInstruct-2/dataset.md) | SFT | cc-by-4.0 | excluded (TSAI ruling) | 20 | 21972791 |
| math | [nvidia__OpenMathReasoning](nvidia__OpenMathReasoning/dataset.md) | SFT + RL prompts | cc-by-4.0 | use, check | 20 | 5678317 |
| math | [open-r1__OpenR1-Math-220k](open-r1__OpenR1-Math-220k/dataset.md) | SFT | apache-2.0 | use, check | 20 | 450258 |
| math | [openai__gsm8k](openai__gsm8k/dataset.md) | eval only (benchmark) | mit | eval only | 20 | 17584 |
| math | [project-numina__NuminaMath-1.5](project-numina__NuminaMath-1.5/dataset.md) | SFT | apache-2.0 | use, check | 20 | 896215 |
| math | [project-numina__NuminaMath-CoT](project-numina__NuminaMath-CoT/dataset.md) | SFT | apache-2.0 | excluded (closed-model generator) | 20 | 859594 |
| math | [zwhe99__DeepMath-103K](zwhe99__DeepMath-103K/dataset.md) | RL (and SFT) | mit | use | 20 | 103022 |
| math, science, coding | [nvidia__Nemotron-Post-Training-Dataset-v1](nvidia__Nemotron-Post-Training-Dataset-v1/dataset.md) | SFT | cc-by-4.0 | excluded (TSAI ruling) | 20 | 25659642 |
| math, science, coding | [nvidia__Nemotron-Post-Training-Dataset-v2](nvidia__Nemotron-Post-Training-Dataset-v2/dataset.md) | SFT | cc-by-4.0 | excluded (TSAI ruling) | 0 | unknown |
| math, science, coding | [open-thoughts__OpenThoughts3-1.2M](open-thoughts__OpenThoughts3-1.2M/dataset.md) | SFT | apache-2.0 | use, check | 20 | 1200000 |
| science | [nvidia__Nemotron-Science-v1](nvidia__Nemotron-Science-v1/dataset.md) | SFT | cc-by-4.0 | use, check; G9 decontamination first | 20 | 226334 |

No sample: `nvidia/Nemotron-Post-Training-Dataset-v2` and `SynthLabsAI/Big-Math-RL-Verified` are gated (the API returns 401 without a Hugging Face login). `aya_dataset` has no Hindi rows in its sample (the Hindi filter failed).

## Excluded after the TSAI ruling (30 Oct)

| Dataset | Reason |
|---|---|
| TIGER-Lab/AceCode-87K | GPT-4o-mini rewrote the questions (OpenAI terms) |
| BhabhaAI/orca-math-word-problems-200k-hindi-filtered | translation of GPT-4-generated Orca-Math (OpenAI terms) |
| GenVRadmin/Aryabhatta-Orca-Maths-Hindi | translation of GPT-4-generated Orca-Math (OpenAI terms) |
| nvidia/OpenMathInstruct-2 | Llama-3.1-405B outputs (Llama licence limits) |
| nvidia/Nemotron-Post-Training-Dataset-v1 | Qwen/DeepSeek teacher outputs, provenance not cleared |
| nvidia/Nemotron-Post-Training-Dataset-v2 | Qwen/DeepSeek teacher outputs, provenance not cleared |

Back in after TSAI's CC-BY-SA update (30 Oct): `nvidia/Nemotron-Math-Proofs-v1` and `nvidia/Nemotron-Math-v2`, now "allowed (CC-BY-SA, attribution needed)". Their traces come from gpt-oss-120b (open-weight, Apache-2.0) and Goedel-Prover-V2-32B.

Also excluded by us under TSAI's rule that data made by closed OpenAI, Anthropic, Google or Llama models stays excluded: `ai4bharat/indic-align` (Llama-2 outputs), `ise-uiuc/Magicoder-OSS-Instruct-75K` (GPT-3.5 instructions) `project-numina/NuminaMath-CoT` (GPT-4 / GPT-4o solutions), `microsoft/rStar-Coder` (GPT-4o synthesized problems and test generators) and `SynthLabsAI/Big-Math-RL-Verified` (Llama-3.1-405B reformulation). These five were not on TSAI's list; ask TSAI to confirm.
