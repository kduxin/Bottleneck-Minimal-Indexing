from typing import List
import os
import argparse
import random
import numpy as np
import time
import warnings
import torch
from torch.utils.data import DataLoader
from collections import defaultdict
from tqdm import tqdm

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy

from .model import T5FineTuner, clean_text
from .nci_transformers import T5Tokenizer
from .data import (
    DocumentRetrievalInferenceDataset,
    DocumentRetrievalInferenceSample,
)
from .index import Index

torch.backends.cuda.matmul.allow_tf32 = True
print(torch.__version__)  # 1.10.0+cu113
print(pl.__version__)  # 1.4.9
logger = None

if "WANDB_API_KEY" not in os.environ:
    warnings.warn(
        "To use wandb, please set WANDB_API_KEY in your environment variables."
    )


def train(args):
    model = T5FineTuner(args)

    if args.ckpt_monitor == "train_loss":
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{args.logs_dir}/{args.data_subdir}",
            filename=args.tag_info + "_{epoch}-{avg_train_loss:.6f}",
            save_on_train_epoch_end=True,
            monitor="avg_train_loss",
            mode="min",
            save_top_k=1,
            every_n_val_epochs=args.check_val_every_n_epoch,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            max_epochs=args.num_train_epochs,
            gradient_clip_val=args.max_grad_norm,
            val_check_interval=args.val_check_interval,
            callbacks=[lr_monitor, checkpoint_callback],
            logger=logger,
        )
    elif args.ckpt_monitor == "recall":
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=f"{args.logs_dir}/{args.data_subdir}",
            filename=args.tag_info + "_{epoch}-{recall1:.6f}",
            monitor="recall1",
            save_on_train_epoch_end=False,
            mode="max",
            save_top_k=1,
            every_n_epochs=args.check_val_every_n_epoch,
        )
        lr_monitor = pl.callbacks.LearningRateMonitor()
        train_params = dict(
            accumulate_grad_batches=args.gradient_accumulation_steps,
            max_epochs=args.num_train_epochs,
            gradient_clip_val=args.max_grad_norm,
            check_val_every_n_epoch=args.check_val_every_n_epoch,
            val_check_interval=args.val_check_interval,
            callbacks=[lr_monitor, checkpoint_callback],
            logger=logger,
        )
    else:
        NotImplementedError("This monitor is not implemented!")

    train_params.update(
        dict(
            accelerator="cuda",
            num_nodes=args.num_nodes,
            devices=args.n_gpu,
            precision="bf16" if args.fp_16 else 32,
            strategy=DDPStrategy(find_unused_parameters=False),
        )
    )

    print("Training ...")
    if args.profiling:
        train_params["profiler"] = "simple"
    trainer = pl.Trainer(**train_params)
    trainer.fit(model, ckpt_path=args.resume_from_checkpoint)


@torch.inference_mode()
def eval_simple(args):
    model = T5FineTuner(args)
    model.eval()

    ckpt_path = args.resume_from_checkpoint
    state_dict = torch.load(ckpt_path)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)

    train_params = dict(
        accumulate_grad_batches=args.gradient_accumulation_steps,
        max_epochs=args.num_train_epochs,
        gradient_clip_val=args.max_grad_norm,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
        val_check_interval=args.val_check_interval,
        logger=logger,
        accelerator="cuda",
        num_nodes=args.num_nodes,
        devices=args.n_gpu,
        precision="bf16" if args.fp_16 else 32,
        strategy=DDPStrategy(find_unused_parameters=False),
    )

    print("Evaluating ...")
    if args.profiling:
        train_params["profiler"] = "simple"
    trainer = pl.Trainer(**train_params)
    trainer.validate(model)


@torch.inference_mode()
def inference(args):
    model = T5FineTuner(args)
    state_dict = torch.load(args.resume_from_checkpoint)
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)
    dataset = DocumentRetrievalInferenceDataset(args)
    compat = dataset.compatibility
    if args.n_test >= 0:
        dataset = [dataset[i] for i in range(args.n_test)]

    device = "cuda"
    model.to(device)
    model.eval()

    loader = DataLoader(
        dataset, batch_size=args.eval_batch_size, shuffle=False, collate_fn=lambda x: x
    )

    results = defaultdict(list)
    for batch in tqdm(loader):
        batch: List[DocumentRetrievalInferenceSample]

        source = tokenizer(
            [clean_text(sample.query) for sample in batch],
            return_tensors="pt",
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
        )

        decode_vocab_size = args.output_vocab_size * args.max_output_length + 2

        outs, scores = model.model.generate(
            source["input_ids"].to(device),
            attention_mask=source["attention_mask"].to(device),
            use_cache=False,
            decoder_attention_mask=None,
            max_length=args.max_output_length,
            num_beams=args.num_return_sequences,
            length_penalty=args.length_penalty,
            num_return_sequences=args.num_return_sequences,
            early_stopping=False,
            decode_embedding=args.decode_embedding,
            decode_vocab_size=decode_vocab_size,
            decode_tree=model.tree.root,
            decoder_index=-1,
            output_scores=True,
        )

        preds_all = [
            Index.from_positioned_index(idseq, kary=args.kary)
            for idseq in outs.tolist()
        ]

        preds_per_sample = [
            preds_all[i : i + args.num_return_sequences]
            for i in range(0, len(preds_all), args.num_return_sequences)
        ]

        for sample, preds in zip(batch, preds_per_sample):
            # hit@1
            hits = []
            for pred in preds:
                if compat.is_compatible(pred, sample.docid):
                    hits.append(True)
                else:
                    hits.append(False)

            for recall_num in args.recall_num:
                hit = any(hits[:recall_num])
                results[f"recall@{recall_num}"].append(1 if hit else 0)

            if any(hits[:100]):
                rank = hits.index(True) + 1
                results["MRR@100"].append(1 / rank)
            else:
                results["MRR@100"].append(0)

    for recall_num in args.recall_num:
        mean_recall = np.mean(results[f"recall@{recall_num}"])
        print(f"recall@{recall_num} = {mean_recall}")

    MRR = np.mean(results["MRR@100"])
    print(f"MRR@100 = {MRR}")


def parsers_parser(raw_args=None):
    parser = argparse.ArgumentParser()

    # Mode
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "eval_simple", "eval"],
        help="""
        train -- training model;
        eval_simple -- evaluate model at training stage, faster while less accurate;
        eval -- evaluate model, slower but more accurate. """,
    )

    # Checkpointing
    parser.add_argument(
        "--load_pretrained_encoder", type=int, default=1, choices=[0, 1]
    )
    parser.add_argument("--logs_dir", type=str, default="logs")

    # Data
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--data_subdir", type=str)
    parser.add_argument(
        "--query_type",
        type=str,
        nargs="*",
        default=["realq", "genq"],
        help="""
        realq -- use ground turth queries;
        genq -- use generated queries;
        ta -- use concatenation of title and abstract as query;
        docseg -- use random segment as query. """,
    )

    # Model specification
    parser.add_argument("--model_name_or_path", type=str, default="t5-")
    parser.add_argument("--tokenizer_name_or_path", type=str, default="t5-")
    parser.add_argument(
        "--model_info",
        type=str,
        default="base",
        choices=["tiny", "mini", "small", "large", "base", "3b", "11b"],
    )
    parser.add_argument("--freeze_encoder", type=int, default=0, choices=[0, 1])
    parser.add_argument("--freeze_embeds", type=int, default=0, choices=[0, 1])

    # Decoder
    parser.add_argument("--decode_embedding", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--dropout_rate", type=float, default=0.1)
    parser.add_argument("--Rdrop", type=float, default=0.15, help="default to 0-0.3")
    parser.add_argument(
        "--Rdrop_only_decoder",
        type=int,
        default=0,
        help="1-RDrop only for decoder, 0-RDrop only for all model",
        choices=[0, 1],
    )
    parser.add_argument("--Rdrop_loss", type=str, default="KL", choices=["KL", "L2"])
    parser.add_argument("--adaptor_decode", type=int, default=1, help="default to 0,1")
    parser.add_argument(
        "--adaptor_efficient", type=int, default=1, help="default to 0,1"
    )
    parser.add_argument("--adaptor_layer_num", type=int, default=4)
    parser.add_argument("--output_vocab_size", type=int, default=10)
    parser.add_argument("--tie_decode_embeddings", type=int, default=1, choices=[0, 1])
    parser.add_argument("--tie_word_embeddings", type=int, default=0, choices=[0, 1])
    parser.add_argument("--gen_method", type=str, default="greedy")
    parser.add_argument("--length_penalty", type=int, default=0.8)

    # Training settings
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--decoder_learning_rate", type=float, default=1e-4)
    parser.add_argument(
        "--ckpt_monitor", type=str, default="recall", choices=["recall", "train_loss"]
    )
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--num_train_epochs", type=int, default=500)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--fp_16", type=int, default=0, choices=[0, 1])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)
    parser.add_argument("--val_check_interval", type=float, default=1.0)

    # Evaluation settings
    parser.add_argument("--tree", type=int, default=1)
    parser.add_argument("--test_set", type=str, default="dev")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument(
        "--recall_num",
        type=list,
        default=[1, 5, 10, 20, 50, 100],
        help="[1,5,10,20,50,100]",
    )
    parser.add_argument(
        "--num_return_sequences",
        type=int,
        default=100,
        help="generated id num (include invalid)",
    )

    # Distributed Training Settings
    parser.add_argument("--n_gpu", type=int, default=1)
    parser.add_argument("--num_nodes", type=int, default=1)

    # Debugging
    parser.add_argument("--test1000", type=int, default=0, help="default to 0,1")
    parser.add_argument("--profiling", type=int, default=0)
    parser.add_argument("--verbose", type=int, choices=[0, 1], default=0)
    parser.add_argument("--n_val", type=int, default=-1)
    parser.add_argument("--n_train", type=int, default=-1)
    parser.add_argument("--n_test", type=int, default=-1)

    # Misc.
    parser.add_argument("--max_input_length", type=int, default=64)
    parser.add_argument("--max_output_length", type=int, default=10)

    args = parser.parse_args(raw_args)

    # Postprocessing of arguments
    args.datadir = f"{args.data_root}/{args.data_subdir}"
    print(f"args.datadir: {args.datadir}")
    args.kary = args.output_vocab_size

    if args.model_info in ["small", "base", "large", "3b", "11b"]:
        args.tokenizer_name_or_path = f"t5-{args.model_info}"
        args.model_name_or_path = f"t5-{args.model_info}"
    elif args.model_info in ["mini", "tiny"]:
        args.tokenizer_name_or_path = f"data/pretrained/t5-{args.model_info}"
        args.model_name_or_path = f"data/pretrained/t5-{args.model_info}"
    else:
        raise ValueError(args.model_info)

    args.gradient_accumulation_steps = max(int(8 / args.n_gpu), 1)

    if args.mode in ["train"]:
        # set to small val to prevent CUDA OOM
        args.num_return_sequences = (
            5  # this would reduce *displayed* performance at training stage
        )
        # args.eval_batch_size = 1
        if args.eval_batch_size * args.num_return_sequences > 400:
            print(
                "Warning: eval_batch_size * num_return_sequences > 400, may cause CUDA OOM"
            )

    if args.model_info == "base":
        args.num_layers = 12
        args.num_decoder_layers = 6
        args.d_ff = 3072
        args.d_model = 768
        args.num_heads = 12
        args.d_kv = 64
    elif args.model_info == "large":
        args.num_layers = 24
        args.num_decoder_layers = 12
        args.d_ff = 4096
        args.d_model = 1024
        args.num_heads = 16
        args.d_kv = 64
    elif args.model_info == "small":
        args.num_layers = 6
        args.num_decoder_layers = 3
        args.d_ff = 2048
        args.d_model = 512
        args.num_heads = 8
        args.d_kv = 64
    elif args.model_info == "mini":
        args.num_layers = 4
        args.num_decoder_layers = 2
        args.d_ff = 1536
        args.d_model = 384
        args.num_heads = 8
        args.d_kv = 64
    elif args.model_info == "tiny":
        args.num_layers = 4
        args.num_decoder_layers = 2
        args.d_ff = 1024
        args.d_model = 256
        args.num_heads = 4
        args.d_kv = 64
    else:
        raise ValueError(args.model_info)

    if args.test1000:
        args.n_train = 100000
        args.n_val = 1000
        args.n_test = 100

    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    args = parsers_parser()
    print(args)
    set_seed(args.seed)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())

    time_str = time.strftime("%Y%m%d-%H%M%S")
    # Note -- you can put important info into here, then it will appear to the name of saved ckpt
    important_info_list = [
        f"kary={args.kary}",
        "-".join(args.query_type),
        args.model_info,
        args.test_set,
        args.ckpt_monitor,
        f"dem={args.decode_embedding}",
        f"ada={args.adaptor_decode}",
        f"adaeff={args.adaptor_efficient}",
        f"adanum={args.adaptor_layer_num}",
        f"RDrop={args.dropout_rate}-{args.Rdrop}-{args.Rdrop_only_decoder}",
    ]

    args.run_name = ".".join(important_info_list)
    if "WANDB_API_KEY" in os.environ:
        logger = WandbLogger(
            name="{}-{}".format(time_str, args.run_name), project="BMI"
        )
    else:
        logger = TensorBoardLogger("logs/")
    ###########################

    args.tag_info = "{}_lre{}d{}".format(
        args.run_name,
        str(float(args.learning_rate * 1e4)),
        str(float(args.decoder_learning_rate * 1e4)),
    )

    if args.mode == "train":
        train(args)
    elif args.mode == "eval_simple":
        assert args.resume_from_checkpoint
        eval_simple(args)
    elif args.mode == "eval":
        assert args.resume_from_checkpoint
        args.recall_num = [1, 5, 10, 20, 50, 100]
        inference(args)
