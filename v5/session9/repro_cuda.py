import sys, os, time
sys.path.insert(0, "src")
import torch
import torch.nn.functional as F
from harness.model import ModelConfig, TinyGPT, OutputHead, count_parameters
from harness.data import get_tokenizer, load_and_tokenize, pack_documents, make_batches
from harness.utils import set_seed, get_device, collect_params
from harness.losses import perplexity

set_seed(1337)
DEVICE = get_device()
enc = get_tokenizer()
VOCAB_SIZE = enc.n_vocab
train_docs = load_and_tokenize(split="train")
print("docs", len(train_docs))

cfg = ModelConfig(vocab_size=VOCAB_SIZE, d_model=256, n_layers=4, n_heads=4, max_seq_len=256)

def make_mtp_batches(packed, seq_len, batch_size):
    n_positions = (len(packed.ids) - 2) // seq_len
    ids = packed.ids[: n_positions * seq_len + 2]
    chunks = []
    for i in range(n_positions):
        s = i * seq_len
        chunks.append(ids[s: s + seq_len + 2])
    chunks = torch.stack(chunks)
    for i in range(0, len(chunks) - batch_size + 1, batch_size):
        block = chunks[i:i + batch_size]
        tok = block[:, :-2]
        tgt1 = block[:, 1:-1]
        tgt2 = block[:, 2:]
        yield tok, tgt1, tgt2

SEQ_LEN=128; BATCH_SIZE=16; N_STEPS=300
set_seed(1337)
mtp_model = TinyGPT(cfg).to(DEVICE)
head_t1 = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=mtp_model.tok_emb.weight).to(DEVICE)
head_t2 = OutputHead(cfg.vocab_size, cfg.d_model, tied_weight=None).to(DEVICE)
opt2 = torch.optim.AdamW(collect_params(mtp_model, head_t1, head_t2), lr=3e-4)

packed_train2 = pack_documents(train_docs, seq_len=SEQ_LEN, n_sequences=BATCH_SIZE*(N_STEPS+2), seed=0)

t0=time.time()
step=0
for tokens_i, t1_i, t2_i in make_mtp_batches(packed_train2, SEQ_LEN, BATCH_SIZE):
    if step>=N_STEPS: break
    tokens_i=tokens_i.to(DEVICE); t1_i=t1_i.to(DEVICE); t2_i=t2_i.to(DEVICE)
    h = mtp_model(tokens_i)
    z1 = head_t1(h); z2 = head_t2(h)
    loss1 = F.cross_entropy(z1.reshape(-1,VOCAB_SIZE), t1_i.reshape(-1))
    loss2 = F.cross_entropy(z2.reshape(-1,VOCAB_SIZE), t2_i.reshape(-1))
    loss_sum = loss1+loss2
    opt2.zero_grad(); loss_sum.backward(); opt2.step()
    if step % 10 == 0:
        print(step, loss1.item(), loss2.item(), time.time()-t0)
    step+=1
print("DONE", step, "steps in", time.time()-t0, "s")
