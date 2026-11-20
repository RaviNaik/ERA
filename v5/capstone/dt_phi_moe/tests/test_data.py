import torch

from dtphi import data
from dtphi.modeling import load_tokenizer


def test_extract_answer_cases():
    assert data.extract_answer("so 3+4=7.\n#### 7") == "7"
    assert data.extract_answer("#### 1,234") == "1234"
    assert data.extract_answer("#### 5\nQ: next question\n#### 99") == "5"  # the first answer counts
    assert data.extract_answer("The answer is \\boxed{12.50}.") == "12.5"
    assert data.extract_answer("first 5 then finally 18 apples") == "18"
    assert data.extract_answer("no digits here") is None
    assert data.is_correct("... #### 72", "blah\n#### 72")
    assert not data.is_correct("... #### 71", "blah\n#### 72")
    assert data.is_correct("#### 72.0", "x\n#### 72")


def test_clean_answer_drops_calculator_annotations_and_keeps_final_line():
    a = "Natalia sold 48/2 = <<48/2=24>>24 clips.\n#### 72"
    assert data.clean_answer(a) == "Natalia sold 48/2 = 24 clips.\n#### 72"


def test_newline_constant_is_the_newline_token(tiny_dir):
    tok = load_tokenizer(str(tiny_dir))
    assert tok.convert_ids_to_tokens(data.NEWLINE_ID) == "<0x0A>"
    assert tok.decode([data.NEWLINE_ID]) == "\n"


def test_splits_are_official_sft_samples(tiny_dir, data_dir):
    tok = load_tokenizer(str(tiny_dir))
    sp = data.Gsm8kSplits(tok, data_dir, seed=0, pool=32, max_len=512)
    s = sp.train_set("gsm8k")
    assert sp.tasks == ["gsm8k"] and len(s) + sp.dropped == 32
    for x in s:
        ids, lab = x["input_ids"], x["label_ids"]
        n_prompt = int((lab == -100).sum())
        assert ids[n_prompt - 1].item() == data.NEWLINE_ID and (x["attention_mask"] == 1).all()
        assert torch.equal(lab[n_prompt:], ids[n_prompt:])
        assert ids[0].item() != tok.bos_token_id  # no BOS, as for the chat template of this model
        text = tok.decode(ids[n_prompt:])
        assert "####" in text and "<<" not in text
        assert len(ids) <= 512
    # the generation prompt of the evaluation is the training prompt
    q = data.load_gsm8k(data_dir, "train")
    pr = data.plain_prompt_ids(tok, q[0]["question"])
    assert pr[-1] == data.NEWLINE_ID


def test_pool_is_a_seeded_subset_and_drops_long_samples(tiny_dir, data_dir):
    tok = load_tokenizer(str(tiny_dir))
    a = data.Gsm8kSplits(tok, data_dir, 0, 20, 512).train_set("gsm8k")
    b = data.Gsm8kSplits(tok, data_dir, 0, 20, 512).train_set("gsm8k")
    c = data.Gsm8kSplits(tok, data_dir, 1, 20, 512).train_set("gsm8k")
    assert all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a, b))
    assert not all(torch.equal(x["input_ids"], y["input_ids"]) for x, y in zip(a, c))
    short = data.Gsm8kSplits(tok, data_dir, 0, 20, 40)
    assert short.dropped > 0 and all(len(x["input_ids"]) <= 40 for x in short.train_set("gsm8k"))
