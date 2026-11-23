import json

from dtphi import report


def _summary(acc, chat, prior):
    return {"gsm8k": {"n": 10, "accuracy": acc, "share_with_####": 0.5},
            "gsm8k_chat": {"n": 10, "accuracy": chat},
            "prior": {"scores": {"hellaswag": prior}, "prior_task_mean": prior, "limit": 50}}


def test_report_has_one_column_per_evaluated_variant(tmp_path):
    for tag, s in (("base", _summary(0.3, 0.5, 0.6)), ("dt", _summary(0.8, 0.7, 0.58)),
                   ("lora", _summary(0.7, 0.65, 0.59))):
        (tmp_path / f"eval_{tag}").mkdir()
        (tmp_path / f"eval_{tag}" / "summary.json").write_text(json.dumps(s))
    res, md = report.build(str(tmp_path))
    header = next(l for l in md.splitlines() if l.startswith("| ") and "base" in l)
    assert "Deletion Tuning" in header and "LoRA" in header and "Full FT" not in header
    row = next(l for l in md.splitlines() if l.startswith("| GSM8K accuracy, chat template"))
    cells = [c.strip() for c in row.strip("|").split("|")]
    assert cells[1:] == ["50.00", "70.00", "65.00", "+20.00", "+15.00"]
    assert res["lora"]["gsm8k_chat"]["accuracy"] == 0.65 and res["ft"] is None


def test_report_with_the_suffix_reads_the_other_folders_and_appends_the_paired_table(tmp_path):
    for tag, s in (("base", _summary(0.3, 0.5, 0.6)), ("dt", _summary(0.8, 0.7, 0.58))):
        (tmp_path / f"eval_{tag}_full").mkdir()
        (tmp_path / f"eval_{tag}_full" / "summary.json").write_text(json.dumps(s))
    (tmp_path / "paired_full.md").write_text("## Paired comparison with the base model (same questions)\n")
    res, md = report.build(str(tmp_path), "_full")
    assert res["dt"]["gsm8k_chat"]["accuracy"] == 0.7 and "Paired comparison" in md
    res0, _ = report.build(str(tmp_path))            # no suffix: those folders do not exist
    assert res0["base"] is None and res0["dt"] is None
