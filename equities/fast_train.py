"""
This script is based on nanoGPT by Karpathy: https://github.com/karpathy/nanoGPT
.... and on the llama2.c repo by Karpathy: https://github.com/karpathy/llama2.c/

This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU small debug run, example:
$ python -m train.py --compile=False --eval_iters=10 --batch_size=8

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 train.py

To run with DDP on 4 gpus across 2 nodes, example:
- Run on the first (master) node with example IP 123.456.123.456:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 --master_addr=123.456.123.456 --master_port=1234 train.py
- Run on the worker node:
$ torchrun --nproc_per_node=8 --nnodes=2 --node_rank=1 --master_addr=123.456.123.456 --master_port=1234 train.py
(If your cluster does not have Infiniband interconnect prepend NCCL_IB_DISABLE=1)
"""

import math
import os
from os.path import dirname, abspath
import sys
import time
import random
import pickle as pkl
from glob import glob
from contextlib import nullcontext
from datetime import datetime
# from functools import partial

ROOT_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(ROOT_DIR)

import numpy as np
import torch
from torch.distributed import destroy_process_group, init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from data_processing.itch_encoding import Vocab, encode_msgs
from fast_model import Transformer, ModelArgs

# additional training options
use_signatures = False
sig_dim = 0
if "--use-signatures" in sys.argv:
    use_signatures = True
    sys.argv.remove("--use-signatures")

# -----------------------------------------------------------------------------
# config values from best sweep run
# I/O
out_dir = "out"
# checkpoint_name = "ckpt_fast_v8.pt"
# checkpoint_name = "ckpt_pretrain_v4.pt"
# checkpoint_name = "ckpt_finetune_AAPL_v4.pt"
checkpoint_name = "ckpt_finetune_SPY_v1.pt"
eval_interval = 50 # 2000
log_interval = 1
eval_iters = 100 # 200
eval_only = False  # if True, script exits right after the first eval
always_save_checkpoint = True # False  # if True, always save a checkpoint after each eval
init_from = "pretrained" # "scratch" # "resume" # "scratch"  # 'scratch' or 'resume' or 'pretrained'
pretrained_checkpoint_name = "ckpt_pretrain_v3.pt"
# wandb logging
wandb_log = True # False # disabled by default
wandb_project = "MarketSimT_fast"
wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
# data
seed = 42
rng = random.Random(seed)
msg_seq_len = 432 # 112 # 432
batch_size = 1 # if gradient_accumulation_steps > 1, this is the micro-batch size
max_seq_len = 10367 # 2687 # 10367 # block_size
use_bpe = False # True # False # if True, use byte pair encoding
use_sink = True # if True, use a dedicated sink token at the start of every training sample (per https://arxiv.org/pdf/2309.17453.pdf)
eom_token_val = 0 # end of message token value
bpe_comp_ratio = 1.56 # bpe compression ratio
bpe_seq_len = int(msg_seq_len * bpe_comp_ratio)
if use_sink:
    max_seq_len += 1
vocab = Vocab()
vocab_size = 12160 # 12544 # 12515
# model
dim = 768 # 1536 # 768
n_layers = 12 # 24 # 12
n_heads = 12 # 16 # 12
n_kv_heads = 12 # 16 # 12
multiple_of = 32
dropout = 0.1 # 0.0 # for pretraining 0 is good, for finetuning try 0.1+
# adamw optimizer
gradient_accumulation_steps = 5 * 8 # 4 # used to simulate larger batch sizes
learning_rate = 1e-3 # max learning rate
max_iters = 4000 # 8000 # 100000 # total number of training iterations
weight_decay = 0.00001
beta1 = 0.9
beta2 = 0.98
grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True  # whether to decay the learning rate
warmup_iters = 50 # 1000  # how many steps to warm up for
# system
device = "cuda"  # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = "bfloat16"  # float32|bfloat16|float16
compile = False # True # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [
    k
    for k, v in globals().items()
    if not k.startswith("_") and isinstance(v, (int, float, bool, str))
]
exec(open("equities/configurator.py").read())  # overrides from command line or config file
config = {k: globals()[k] for k in config_keys}  # will be useful for logging
# -----------------------------------------------------------------------------

# fixing some hyperparams to sensible defaults
lr_decay_iters = max_iters  # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # 0.0 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

# various inits, derived attributes, I/O setup
ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
if ddp:
    init_process_group(backend="nccl")
    ddp_rank = int(os.environ["RANK"])
    ddp_local_rank = int(os.environ["LOCAL_RANK"])
    ddp_world_size = int(os.environ["WORLD_SIZE"])
    device = f"cuda:{ddp_local_rank}"
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0  # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank  # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * max_seq_len
if master_process:
    print(f"tokens per iteration will be: {tokens_per_iter:,}")
    print(f"breaks down as: {gradient_accumulation_steps} grad accum steps * {ddp_world_size} processes * {batch_size} batch size * {max_seq_len} max seq len")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(seed + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
device_type = "cuda" if "cuda" in device else "cpu"  # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
ctx = (
    nullcontext()
    if device_type == "cpu"
    else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
)

# locate directories with training and validation data
# train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/train/')
# train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/five_assets/train/')
# train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/pre_train/')
# train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/fine_tune/AAPL/train/') # same AAPL data that was used for pre-training (only exclusive AAPL data)
train_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/fine_tune/SPY/train/')
train_message_files = sorted(glob(str(train_data_dir) + '/*message*.npy'))
assert len(train_message_files) > 0, f'no message files found in {train_data_dir}'
# val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/val/')
# val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/five_assets/val/')
# val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/pre_train/val/')
# val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/fine_tune/AAPL/val/') # this was not used in pre-training
val_data_dir = os.path.join(os.path.abspath(''), 'dataset/proc/ITCH/multi/fine_tune/SPY/val/')
val_message_files = sorted(glob(str(val_data_dir) + '/*message*.npy'))
assert len(val_message_files) > 0, f'no message files found in {val_data_dir}'
# fill list with datasets, seperated by file (each file is a day of data)
train_datasets = []
for file in train_message_files:
    train_datasets.append(np.load(file, mmap_mode='r'))
val_datasets = []
for file in val_message_files:
    val_datasets.append(np.load(file, mmap_mode='r'))
# BPE encoding
if use_bpe:
    # load the tokenizer
    # with open('tokenizer/bpe_tokenizer.pkl', 'rb') as f:
    with open('tokenizer/bpe_tokenizer_pretrain.pkl', 'rb') as f:
        bpe_tokenizer = pkl.load(f)
# verify sink token compatibility with dataloader implementation
if use_sink:
    assert vocab.SINK_TOK == 1
    # assert max_seq_len == 10368
# poor man's data loader
def get_batch_raw(split):
    # data = train_data if split == 'train' else val_data
    datasets = train_datasets if split == 'train' else val_datasets
    data = rng.choice(datasets)
    ix = torch.randint(len(data) - msg_seq_len, (batch_size,))
    if use_bpe:
        assert batch_size == 1, "batch size must be 1 for BPE encoding (for now)" # TODO: make this batch-wise
        # basic encoding of the messages
        # basic_encoded = [(encode_msgs((data[i:i+msg_seq_len]).astype(np.int64), vocab.ENCODING)) for i in ix]
        basic_encoded = [(encode_msgs((data[i:i+bpe_seq_len]).astype(np.int64), vocab.ENCODING)) for i in ix]
        # bpe encode the messages and concat EOM token to the end
        bpe_encoded = []
        for batch in range(len(basic_encoded)):
            for msg in range(len(basic_encoded[batch])):
                encoded_msg = bpe_tokenizer.bpe_encode(basic_encoded[batch][msg]) + [eom_token_val]
                if (len(encoded_msg) + len(bpe_encoded)) <= (msg_seq_len * 24):
                    bpe_encoded = bpe_encoded + encoded_msg
                else:
                    break
                # bpe_encoded = bpe_encoded + bpe_tokenizer.bpe_encode(basic_encoded[batch][msg]) + [eom_token_val]
        # convert to tensor, unsqueeze to add batch dimension
        x = torch.tensor(bpe_encoded).unsqueeze(0)
    else:
        # use basic encoding only
        x = torch.stack([torch.from_numpy((encode_msgs((data[i:i+msg_seq_len]).astype(np.int64), vocab.ENCODING)).reshape(-1)) for i in ix])
    if use_sink:
        # append sink token to start of each batch sequence (since vocab.SINK_TOK = 1, we can just use torch.ones)
        x = torch.cat([torch.ones((batch_size, 1), dtype=torch.int), x], dim=1)
    # target y is the same as x but shifted by one token
    y = x[:, 1:]
    y = y.type(torch.LongTensor) # casting to long for cross entropy loss fn
    x = x[:, :-1] # offset x by one (final) token to match y
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

if use_signatures:
    from equities.signature_dataset import SignatureDataset

    signature_dataset = SignatureDataset(get_batch_raw, sig_dim=sig_dim)
    sig_dim = signature_dataset.sig_dim
    config["sig_dim"] = sig_dim
    config["use_signatures"] = use_signatures

    def get_batch(split):
        tokens, sigs, targets = signature_dataset.get_batch(split)
        return tokens, sigs, targets
else:
    def get_batch(split):
        tokens, targets = get_batch_raw(split)
        return tokens, None, targets
    config["sig_dim"] = sig_dim
    config["use_signatures"] = use_signatures

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# model init
model_args = dict(
    dim=dim,
    n_layers=n_layers,
    n_heads=n_heads,
    n_kv_heads=n_kv_heads,
    vocab_size=vocab_size,
    multiple_of=multiple_of,
    max_seq_len=max_seq_len,
    dropout=dropout,
    sig_dim=sig_dim,
    use_signatures=use_signatures,
)  # start with model_args from command line
if init_from == "scratch":
    # init a new model from scratch
    print("Initializing a new model from scratch")
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
elif init_from == "resume":
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len", "sig_dim", "use_signatures"]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint["iter_num"]
    best_val_loss = checkpoint["best_val_loss"]
elif init_from == "pretrained":
    print(f"Finetuning from {out_dir}")
    # finetuning from a checkpoint.
    ckpt_path = os.path.join(out_dir, pretrained_checkpoint_name)
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint["model_args"]
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len", "sig_dim", "use_signatures"]:
        if k in checkpoint_model_args:
            model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = ModelArgs(**model_args)
    model = Transformer(gptconf)
    state_dict = checkpoint["model"]
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = "_orig_mod."
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    # iter_num = checkpoint["iter_num"]
    # best_val_loss = checkpoint["best_val_loss"]

    # # freeze the first 6 layers
    # for param in model.transformer[:6].parameters():
    #     param.requires_grad = False
    # # unfreeze the last 6 layers
    # for param in model.transformer[6:].parameters():
    #     param.requires_grad = True
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == "float16"))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == "resume" and "optimizer" in checkpoint:
    optimizer.load_state_dict(checkpoint["optimizer"])
checkpoint = None  # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model)  # requires PyTorch 2.0

# wrap model into DDP container
if ddp:
    # Ignore the `freqs_cis` buffer so that DDP does not broadcast it at
    # construction time since NCCL does not support `ComplexFloat`
    prefix = "_orig_mod." if compile else ""
    model._ddp_params_and_buffers_to_ignore = {prefix + "freqs_cis"}
    model = DDP(model, device_ids=[ddp_local_rank])

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)  # keep on CPU
        for k in range(eval_iters):
            X, sigs, Y = get_batch(split)
            with ctx:
                logits = model(X, sigs=sigs, targets=Y)
                loss = raw_model.last_loss
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# training loop
X, sigs, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0  # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model  # unwrap DDP container if needed
running_mfu = -1.0
print("Training is starting.")
start_time = time.time()
while True:
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            try:
                wandb.log(
                    {
                        "iter": iter_num,
                        "tokens": iter_num * tokens_per_iter,
                        "loss/train": losses["train"],
                        "loss/val": losses["val"],
                        "lr": lr,
                        "mfu": running_mfu * 100,  # convert to percentage
                    }, step = iter_num
                )
            except Exception as e:
                print(f"logging to wandb failed: {e}")
        if losses["val"] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses["val"]
            if iter_num > 0:
                checkpoint = {
                    "model": raw_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "model_args": model_args,
                    "iter_num": iter_num,
                    "best_val_loss": best_val_loss,
                    "config": config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, checkpoint_name))
    if iter_num == 0 and eval_only:
        break

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        if ddp:
            # in DDP training we only need to sync gradients at the last micro step.
            # the official way to do this is with model.no_sync() context manager, but
            # I really dislike that this bloats the code and forces us to repeat code
            # looking at the source of that context manager, it just toggles this variable
            model.require_backward_grad_sync = micro_step == gradient_accumulation_steps - 1
        with ctx:
            logits = model(X, sigs=sigs, targets=Y)
            loss = raw_model.last_loss
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, sigs, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float, scale up due to the divide above. note: this is a CPU-GPU sync point
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5:  # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu
        print(
            f"{iter_num} | loss {lossf:.4f} | lr {lr:e} | {dt*1000:.2f}ms | mfu {running_mfu*100:.2f}%"
        )
    iter_num += 1
    local_iter_num += 1

    # termination conditions
    if iter_num > max_iters:
        break

end_time = time.time()
print(f"Training is done. Took {(end_time-start_time)/60:.2f} minutes.")

if ddp:
    destroy_process_group()