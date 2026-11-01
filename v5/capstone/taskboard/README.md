# ERA V5 Capstone Task Board

A task board for the V5 capstone: individual work (`code-go`), the G10 team tasks, the G12 rotation, dependencies on other teams, and the lab milestones.

All data lives in one file, **`tasks.json`**. The board loads it, and saves every change back to it, through a small local server (`server.py`). There is no browser storage for data, so the file in this folder is always the single source of truth, and it is what you commit to git.

## Start it

```bash
cd v5/capstone/taskboard
python3 server.py
```

It opens Chrome (your default browser) at <http://127.0.0.1:8765/>. Stop it with Ctrl+C. Python 3.8 or newer; nothing to install.

Options: `--no-open` (do not open a browser), `--port 9000`, `--data /path/to/other.json`.

Opening `index.html` directly (double-click) does not work: a web page cannot write files, so the board shows how to start the server.

## Where the data is and how it is protected

| What | Where |
|---|---|
| Live data | `taskboard/tasks.json` (pretty-printed, one change per line in git diffs) |
| Backups | `taskboard/backups/tasks-YYYYMMDD-HHMMSS.json`: a copy of the previous file before a save (at most one every 5 minutes, newest 40 kept), and always before an import or restore. Not committed (`.gitignore`) |
| Git | `tasks.json` is the version-controlled backup: `git add v5/capstone/taskboard/tasks.json && git commit` |

- Every save goes straight to `tasks.json` (written to a temp file, then renamed, so a crash cannot leave half a file). The header shows **Saved to tasks.json**, **Saving...** or **Not saved**; click it to retry.
- `tasks.json` has a revision number. If the file changes outside the page (a `git pull`, a text editor, a second tab) the board reloads it when you come back to the tab, or, if you have unsaved edits, asks whether to load the file or save yours over it. The old file is backed up first.
- **Data menu:** Reload from tasks.json, Restore a backup (lists the backups with date and task count), Download a copy, and Replace tasks.json from a file.
- The server listens on `127.0.0.1` only, checks the `Host` and `Origin` headers, and validates the data (unique IDs, known status and priority) before it writes anything.

## What you can do

| Feature | How |
|---|---|
| Add a task | **+ New task** (or press `n`) |
| Edit a task | Click a card or row to open the side panel |
| Mark complete / pending | Tick the box on a card, drag the card to another column, or change **Status** |
| Statuses | Pending, In progress, In review, Blocked, Complete |
| Deadline | Date and time (IST, default 23:59). Overdue and due-in-2-days show in red and amber |
| Weekly / daily tasks | **Type = Recurring**, then **Repeats = daily / weekly**; **Week** is the lab week (1 to 4) |
| Assignees | Several per task; empty counts as **Unassigned** |
| PR / commit numbers | **PRs, commits and links**: kind, repo, number or sha. Each becomes a GitHub link |
| Comments | Dated notes on each task: decisions, blockers, links |
| Blocked by | A task ID (for example `T-060`) or free text |
| Workstreams | Pick one, or type a new name in the task panel |
| Find things | Search (title, notes, labels, PR or commit), filters for workstream, status, assignee, week, type, **Mine**, **Hide complete** |
| Views | **Board** (drag and drop), **List** (sortable), **Schedule** (Overdue, Today, Next 7 days, Later) |
| Summary | The tiles at the top are shortcuts (Open, Overdue, Blocked, Mine, Unassigned, Complete) |

Only the light/dark choice is kept in the browser. Repos used for PR and commit links are named in `tasks.json` under `repos` (`lab`, `fork`, `mine`, `era`); add more there.

## tasks.json format

`{ rev, updated, me, repos, workstreams, tasks }`. A task has `id`, `type` (task, recurring, milestone), `workstream`, `title`, `desc`, `status` (`todo`, `progress`, `review`, `blocked`, `done`), `priority` (`high`, `medium`, `low`), `assignees`, `due`, `dueTime`, `week`, `recurring`, `labels`, `blockedBy`, `refs` (`kind`: pr, commit, issue, comment, link), `comments`. `rev` and `updated` are set by the server on each save. You can also edit the file by hand; the board picks the change up.

## Keep it public-safe

This repo is public. Put only public handles, public PR and commit numbers and public deadlines in the board. Do not add emails, keys, tokens, or the text of private messages (for example direct messages from TSAI): note the action instead.

## Seed content

Revision 1 (1 Nov 2026) holds 47 items from the lab posts and our own work: the `code-go` diagnosis (PR #36), the G10 week 2 plan (tasks A to G and their due dates), the G12 rotation task, the Data A gold-set dependency (lab PR #140), and the Gate 1 and Gate 2 dates. The G10 member letters (A to G) are not assigned yet, so those tasks start unassigned.
