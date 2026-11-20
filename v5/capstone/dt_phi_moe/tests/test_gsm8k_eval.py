from dtphi.eval_gsm8k import ANSWER_DONE, trim_to_answer
from dtphi.modeling import load_tokenizer


def test_answer_done_needs_the_number_to_be_finished():
    assert not ANSWER_DONE.search("so we get #### 7")
    assert not ANSWER_DONE.search("so we get ####")
    assert ANSWER_DONE.search("so we get #### 72\n")
    assert ANSWER_DONE.search("#### 1,234.5 ")
    assert not ANSWER_DONE.search("#### 12")


def test_trim_cuts_right_after_the_first_complete_answer(tiny_dir):
    tok = load_tokenizer(str(tiny_dir))
    text = "Step one.\n#### 42\nQ: something else\n#### 99\n"
    ids = tok(text, add_special_tokens=False)["input_ids"]
    cut = trim_to_answer(tok, ids)
    assert tok.decode(cut).startswith("Step one.\n#### 42") and "99" not in tok.decode(cut)
    assert len(cut) < len(ids)
    no_answer = tok("just words", add_special_tokens=False)["input_ids"]
    assert trim_to_answer(tok, no_answer) == no_answer
