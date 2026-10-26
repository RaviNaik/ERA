# code-go data diagnosis

Per-document review of the 500-row `code-go` pack (StarCoder Go).

- `labels.csv`, `medicine.md`: the submitted deliverables.
- `review/NNN_<doc_id>.go` with `review/NNN_<doc_id>.diagnosis.md`: each document next to its diagnosis and the medicine result. `INDEX.md` lists all 500.
- Three documents (049 ransomware source, 141 hardcoded secret, 493 personal email) have their diagnosis file only; the `.go` text is left out on purpose.
