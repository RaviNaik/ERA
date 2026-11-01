# ERA V5 Capstone Task Board

A single-page task board for the V5 capstone: individual work (`code-go`), the G10 team tasks, the G12 rotation, dependencies on other teams, and the lab milestones. It needs no install, no build and no server.

## Open it

Open `index.html` in a browser (double-click it). `tasks.js` must stay next to it.

Your edits are saved in the browser (`localStorage`), so they survive a refresh but stay on one browser. To keep them in git, use **Data > Export tasks.js**, replace `tasks.js` with the downloaded file and commit it.

## What you can do

| Feature | How |
|---|---|
| Add a task | **+ New task** (or press `n`) |
| Edit a task | Click a card or row to open the side panel |
| Mark complete / pending | Tick the box on a card, drag the card to another column, or change **Status** |
| Statuses | Pending, In progress, In review, Blocked, Complete |
| Deadline | Date and time (IST, default 23:59). Overdue and due-in-2-days show in red and amber |
| Weekly / daily tasks | **Type = Recurring**, then **Repeats = daily / weekly**; **Week** is the lab week (1 to 4) |
| Assignees | Several per task; leave empty to see it in the **Unassigned** count |
| PR / commit numbers | **PRs, commits and links**: kind, repo, number or sha. Each becomes a GitHub link |
| Comments | Dated notes on each task: decisions, blockers, links |
| Blocked by | A task ID (for example `T-060`) or free text |
| Find things | Search (title, notes, labels, PR or commit), plus filters for workstream, status, assignee, week, type, **Mine** and **Hide complete** |
| Views | **Board** (columns, drag and drop), **List** (sortable table), **Schedule** (Overdue, Today, Next 7 days, Later) |
| Summary | The tiles at the top are clickable shortcuts (Open, Overdue, Blocked, Mine, Unassigned, Complete) |

Repos for PR and commit links are named in `tasks.js` under `repos`: `lab` (the TSAI lab), `fork` (Sai Teja's fork), `mine` (my lab fork) and `era` (this repo). Add more there.

## Data

`tasks.js` holds `window.TASKBOARD_SEED = { rev, updated, me, repos, tasks }`.

- `rev` goes up each time the file is exported. If `tasks.js` in the repo is newer than the copy in your browser, the board shows a banner with two choices: merge the new tasks and keep your edits, or keep yours.
- **Data > Export JSON backup** saves a backup; **Import** loads a board export from this tool. **Reset** discards browser edits and reloads `tasks.js`.
- A task: `id`, `type` (task, recurring, milestone), `workstream`, `title`, `desc`, `status` (`todo`, `progress`, `review`, `blocked`, `done`), `priority`, `assignees`, `due`, `dueTime`, `week`, `recurring`, `labels`, `blockedBy`, `refs` (`kind`: pr, commit, issue, comment, link), `comments`.

## Keep it public-safe

This repo is public. Put only public handles, public PR and commit numbers and public deadlines here. Do not add emails, keys, tokens, or the text of private messages (for example direct messages from TSAI): note the action instead.

## Seed content

The first version (`rev` 1, 1 Nov 2026) holds 47 items taken from the lab posts and our own work: the `code-go` diagnosis (PR #36), the G10 week 2 plan (tasks A to G with their due dates), the G12 rotation task, the Data A gold-set dependency (lab PR #140), and the Gate 1 and Gate 2 dates. The G10 member letters (A to G) are not assigned yet, so those tasks start unassigned.
