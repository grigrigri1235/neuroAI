# Workflow: how we work on this project (lambda + Newton + Claude Code)

Written 2026-10-15, from the conversation where we re-ran Coconut on GPT-2 and moved on to Qwen.
For: the user, and any future Claude/agent session on this repo. Read together with
`KNOWLEDGE_RETENTION.md` (the *what* of the project) and `~/.claude/CLAUDE.md` (the global rules).

---

## 1. The setup in one picture

```
            write code, plans, docs                    run everything (GPU)
   ┌──────────────────────────────┐            ┌──────────────────────────────┐
   │ lambda                        │  git push  │ Newton (Technion CS Slurm)    │
   │ - Claude Code runs here       │ ─────────► │ - user runs all commands      │
   │ - repo /home/eran.b/neuroAI   │   GitHub   │ - repo /home/eran.b/neuroAI   │
   │ - no experiments run here     │ ◄───────── │ - GPU: L40S 47GB (Slurm name  │
   │ - Claude reads logs/results   │  git push  │   "L40"), conda, HF cache     │
   └──────────────────────────────┘ (results)  └──────────────────────────────┘
```

- **lambda**: Claude Code works here, because it has issues on Newton. Claude writes code, plans and docs, and
  reads logs and results. Nothing heavy runs on lambda.
- **Newton**: all GPU work. Same paths as lambda (`/home/eran.b/neuroAI`). The user runs every command there.
- **GitHub** (`git@github.com:grigrigri1235/neuroAI.git`, SSH): the only bridge. Code goes lambda → Newton. Logs and
  results go Newton → lambda.
- **Claude never runs bash.** Global rule 5. Claude only reads and writes files, and gives the user exact commands.

---

## 2. How a conversation goes

### 2.1 Every message
- **Starts with a recap of the global rules** (`~/.claude/CLAUDE.md`, rule 9). It also says how Claude might break
  them in this step and what it does to avoid that (e.g. "I could take 'i trust you' as approval; I'll ask once more").
- **Short, simple language** (rule 4). Tables for numbers. No hype words ("proves", "decisive", "100%").

### 2.2 The lifecycle of a piece of work

| Step | What happens | Who | Rule |
|---|---|---|---|
| 1. Read | Read `KNOWLEDGE_RETENTION.md` first, then the paper / code needed. Read-only tools only. Checking a model on the HF Hub (file list, config) is also read-only and allowed. | Claude | |
| 2. Plan | Write a plan file in `docs/planning/<name>.md`, not in chat. Include what reading the code revealed (e.g. "the official `Coconut.forward` crashes on two latent blocks"). | Claude | 2, Phase 1 |
| 3. STOP | Show a short summary + the plan link + the open questions. Wait for an explicit "proceed". "I trust you" or "maybe we should" is not a proceed: ask once more. | Claude → user | 3 |
| 4. Build | Split into tiny numbered parts. After each part write only `done with part i`, and keep going to the end. | Claude | 1, 6 |
| 5. Document | Update the plan status, `docs/requirment/codebase_changes.md` (every code change), `KNOWLEDGE_RETENTION.md` (status, findings), and memory. | Claude | |
| 6. Commands | Give exact commands, grouped by machine ("On lambda", "On Newton"), to paste one line at a time. Say what output to expect after each check. | Claude | 5 |
| 7. Run | Run the commands, watch the job, push logs/results back, `git pull` on lambda, say "pulled". | user | 8 |
| 8. Read results | Read the result files with the Read tool (never bash). Report numbers + CIs, what can and cannot be claimed, and a recommendation. | Claude | |
| 9. Decide | The user decides the next step. Big changes (a pivot, a new model) go back to step 2. | user | 3 |

### 2.3 When something goes wrong
- **Crash or failed check:** Claude reports the reason, from the logs, and stops (rule 7). No silent fixes.
- **Small fix with clear evidence** (e.g. `sinfo` shows the GPU is called `L40`, not `L40S`): Claude explains
  the cause, edits the file, and gives the push/pull/resubmit commands.
- **Bad or empty result** (e.g. floor effect): Claude says plainly what it means and what it does not mean. It
  offers options with one recommendation. It does not oversell a weak result ("you're right, it's mostly a
  confirmation").
- **The user insists on an idea** (e.g. interleaved latent thinking): Claude finds a way to really test it (here,
  a better model) behind a gate check, instead of claiming an untested result.

### 2.4 Approval words
- **Approval:** "proceed", "proceed with the plan", "I approve the pivot and your plan", "lets go", a direct
  order ("change the mail to X", "document it in workflow.md").
- **Not approval:** "maybe we should ...", "i trust you" (on its own), a question.

---

## 3. The lambda ↔ Newton loop (exact commands)

### 3.1 Code goes to Newton
On **lambda** (the user runs these; Claude only gives them):
```bash
cd /home/eran.b/neuroAI
git add <the files Claude names>
git commit -m "<message>" -m "Co-Authored-By: Claude ... <noreply@anthropic.com>"
git push
```
On **Newton**:
```bash
cd /home/eran.b/neuroAI
git pull
```

### 3.2 Submit a job on Newton
```bash
cd /home/eran.b/neuroAI                      # always from the repo root (slurm -o/-e paths are relative)
conda activate neuroAI_project
sbatch --test-only coconut/experiments/<job>.slurm   # checks the request, runs nothing
sbatch coconut/experiments/<job>.slurm               # prints "Submitted batch job <JOBID>"
squeue -u eran.b                             # PD = waiting for a GPU, R = running
tail -f logs/out_job<JOBID>.txt              # progress; Ctrl+C stops watching, not the job
```
Cancel: `scancel <JOBID>`. How it ended: `sacct -j <JOBID> --format=JobID,State,Elapsed,ExitCode`.

### 3.3 Results come back to lambda
On **Newton** (pass or fail, always push, so Claude can read the logs):
```bash
cd /home/eran.b/neuroAI
git add logs coconut/results
git commit -m "<what ran>: logs and results"
git push
```
On **lambda**: `cd /home/eran.b/neuroAI && git pull`, then tell Claude "pulled". Claude then reads, for example:
`logs/out_job<id>.txt`, `logs/<step>_<id>.log`, `coconut/results/<model>/results.json`.

---

## 4. Newton specifics

- **Login:** `ssh -X eran.b@132.68.39.200` (Technion network or VPN). The prompt must say `eran.b@newton`. Running a
  check on lambda by mistake gives "file not found" (data and models only exist on Newton).
- **Lab:** `houdini (LMM)`. Jobs go to the default partition `public`. Slurm picks any free node with the requested
  GPU (e.g. `newton5`, `bruno4`). No `--partition`/`--account` is needed.
- **GPU name:** the L40S 47GB is called **`L40`** in Slurm: `--gres=gpu:L40:1`. `gpu:L40S:1` fails with
  "Requested node configuration is not available". Check names with `sinfo -N -o "%N %G %c %m %l %P %T"`.
- **Slurm template** (Technion format, kept in every `.slurm` file):
  `--cpus-per-task=8`, `--gres=gpu:L40:1`, `--mem`, `--time`, `--mail-user=eran.b@campus.technion.ac.il`,
  `--mail-type=ALL`, `--job-name`, `-o ./logs/out_job%j.txt`, `-e ./logs/err_job%j.txt`, then `module purge`,
  `source /home/eran.b/miniconda3/etc/profile.d/conda.sh`, `conda activate neuroAI_project`.
- **Logs:** `/home/eran.b/neuroAI/logs/` (must exist before `sbatch`: `mkdir -p logs`). Each python step writes its
  own log there. Bash output always goes to a file (rule 8).
- **Offline jobs:** jobs set `HF_HUB_OFFLINE=1`, because compute nodes may have no internet. Every model is downloaded
  first on the login node with `snapshot_download(...)`. For big repos, only the needed files (`allow_patterns`).
- **Environment** (done once, 2026-10-15): Miniconda in `/home/eran.b/miniconda3`; Anaconda ToS accepted with
  `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main` (and `/pkgs/r`); env from
  `environment_newton.yml` (python 3.12, torch 2.6.0, transformers 4.52.4, datasets 3.6.0). Data:
  `cd coconut && bash preprocessing/gsm_icot.bash`. GPT-2 + checkpoint: `python experiments/common.py` ("loaded OK").

---

## 5. How experiments are built

- **Official Coconut files are never edited** (`coconut.py`, `run.py`, `dataset.py`, `utils.py`). New code lives in
  `coconut/experiments/` and imports them.
- **Gate first, then the big run.** Every new model or pipeline first gets a short check job that tests that it
  loads right, that our loop matches the official/HF code, and that the latents do something. Only if it passes
  does the full run get planned and approved.
  Examples: `smoke_test.py` (C0-C8, GPT-2), `qwen_check.py` (Qwen2.5-7B, failed: garbage output),
  `qwen3_check.py` (Qwen3-4B, G0-G3).
- **The smoke test is inside the job** and gates it: if a required check fails, nothing else runs.
- **Controls that remove one thing:** S5 (same latent slots, no feedback) vs S3 (feedback), S4 (no slots).
  Plain-embedding control in the native format (`run_native.py --fill plain`).
- **Paired stats on the full test set** (GSM8K 1319): accuracy + Wilson CI, exact McNemar, paired bootstrap CI.
  A difference counts only if its CI excludes 0 (`score.py`).
- **Everything saved per example** (prompt length, raw/cut output, prediction, correct, time), plus run metadata
  (versions, GPU, job id). Results in `coconut/results/<model>/`.
- **Honest framing:** say what was tested and what was not. Example: the GPT-2 control shows that the gain needs the
  hidden-state feedback **in the model's own format**. It does **not** test interleaved latent thinking.
- **Low-level technical risks** (config formats, tokenizer versions) are handled by checks in code and noted in
  one line. Results and claims are explained in full (oral check).

---

## 6. Pitfalls we hit (and the fix)

| What happened | Fix |
|---|---|
| `conda env create` → `CondaToSNonInteractiveError` | accept the ToS (the user chose this) or use `--override-channels -c conda-forge` |
| `sbatch` → "Requested node configuration is not available" | the GPU is `L40` in Slurm, not `L40S` |
| ran a Newton check on lambda → file not found | check the prompt / `hostname` says newton |
| pasted a multi-line block → `'{echo' not found` | paste commands one line at a time |
| Claude found no results | they weren't pushed from Newton / pulled on lambda yet |
| official `Coconut.forward` crashes with two latent blocks (IndexError) | our own `fill_latents()` (checked equal on one block) |
| GPT-2 ignores the few-shot step format (floor effect) | test in the native format; move the interleaving test to a bigger model |
| Qwen2.5-7B export writes garbage even without latents | gate check caught it; abandoned |
| Qwen3-4B export saved by transformers 5.x (nested `rope_theta`) | loader copies it and checks the rotary frequencies |

---

## 7. Checklists

**Start of a conversation (Claude):** read `KNOWLEDGE_RETENTION.md`, this file, memory; recap the rules; check what
is already on lambda (results pulled?) with read-only tools.

**Before `sbatch` (user):** on Newton (`hostname`); `git pull` done; `logs/` exists; models prefetched;
`sbatch --test-only` OK.

**After a job (user → Claude):** `sacct` / `tail` of `out_job`; push `logs` + `coconut/results`; `git pull` on
lambda; tell Claude "pulled".

**After results (Claude):** read the logs and results JSON; report numbers + CIs + claims/non-claims + a
recommendation; update `KNOWLEDGE_RETENTION.md` once the user decides.
