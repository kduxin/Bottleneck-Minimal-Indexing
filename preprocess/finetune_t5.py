import os
import time
from argparse import ArgumentParser
import pandas as pd
import torch
import warnings
from torch.utils.data import DataLoader, Dataset
import transformers
from lightning import pytorch as pl
from lightning.pytorch.loggers import WandbLogger

torch.set_float32_matmul_precision("high")

if "WANDB_API_KEY" in os.environ:
    warnings.warn(
        "To use wandb, please set WANDB_API_KEY in your environment variables."
    )


class T5FineTunerSimple(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.save_hyperparameters(args)

        self.model = transformers.T5ForConditionalGeneration.from_pretrained(
            self.args.raw_ckpt
        )
        self.tokenizer = transformers.T5Tokenizer.from_pretrained(self.args.raw_ckpt)

    def forward(self, input_ids, attention_mask, labels):
        loss = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        ).loss
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("train_loss", loss, on_epoch=True)
        return loss

    def train_dataloader(self):
        print("Loading training data...")
        dataset = Doc2QueryDataset(
            self.args.train_data_path,
            self.tokenizer,
            doc_max_len=self.args.doc_max_len,
            query_max_len=self.args.query_max_len,
            test1000=self.args.test1000,
        )
        dataloader = DataLoader(
            dataset,
            shuffle=True,
            batch_size=self.args.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=4,
        )
        print("Data loading finished.")
        return dataloader

    def validation_step(self, batch, batch_idx):
        loss = self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        self.log("validation_loss", loss, on_epoch=True, sync_dist=True)
        return loss

    def val_dataloader(self):
        print("Loading val data...")
        dataset = Doc2QueryDataset(
            self.args.val_data_path,
            self.tokenizer,
            doc_max_len=self.args.doc_max_len,
            query_max_len=self.args.query_max_len,
            test1000=self.args.test1000,
        )
        dataloader = DataLoader(
            dataset,
            shuffle=False,
            batch_size=self.args.batch_size,
            collate_fn=dataset.collate_fn,
            num_workers=4,
        )
        print("Data loading finished.")
        return dataloader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            weight_decay=self.args.weight_decay,
            lr=self.args.lr,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.args.lr,
            total_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]


class Doc2QueryDataset(Dataset):
    def __init__(
        self,
        tsv_file,
        tokenizer,
        doc_max_len=2**20,
        query_max_len=2**20,
        test1000: bool = False,
    ):
        def _load_samples():
            df = pd.read_csv(
                tsv_file,
                usecols=['doc', 'query'],
                sep="\t",
                chunksize=1000 if test1000 else None,
            )
            if test1000:
                df = next(df)
            return list(zip(df["doc"], df["query"]))

        self.samples = _load_samples()
        self.tokenizer = tokenizer
        self.doc_max_len = doc_max_len
        self.query_max_len = query_max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        doc, query = self.samples[i]
        return doc, query

    def collate_fn(self, batch):
        docs, queries = zip(*batch)
        inputs = self.tokenizer(
            docs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.doc_max_len,
        )
        labels = self.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return {
            "input_ids": inputs.input_ids,
            "attention_mask": inputs.attention_mask,
            "labels": labels,
        }


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--raw_ckpt", type=str)
    parser.add_argument("--finetuned_ckpt", type=str)
    parser.add_argument("--train_data_path", type=str)
    parser.add_argument("--val_data_path", type=str)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--doc_max_len", type=int, default=512)
    parser.add_argument("--query_max_len", type=int, default=64)
    parser.add_argument("--test1000", type=int, choices=[0, 1], default=0)
    parser.add_argument("--num_nodes", type=int, default=1)
    args = parser.parse_args()

    args.test1000 = bool(args.test1000)
    return args


def train(args):
    finetuner = T5FineTunerSimple(args)

    run_info = ".".join(
        [
            f"{args.raw_ckpt.replace('/', '_')}",
            f"{args.train_data_path.replace('/', '_')}",
            f"ep={args.epochs}",
            f"lr={args.lr}",
            f"dmaxlen={args.doc_max_len}",
            f"qmaxlen={args.query_max_len}",
        ]
    )

    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=f"{args.finetuned_ckpt}",
        filename=run_info + ".{epoch}.{validation_loss:.6f}",
        save_top_k=1,
        monitor="validation_loss",
        mode="min",
    )
    time_str = time.strftime("%Y%m%d-%H%M%S")
    wandb_logger = WandbLogger(project="docT5query", name=f"{time_str}.{run_info}")

    trainer = pl.Trainer(
        callbacks=[lr_monitor, checkpoint_callback],
        max_epochs=args.epochs,
        logger=wandb_logger,
        num_nodes=args.num_nodes,
    )

    trainer.fit(finetuner)

    if trainer.is_global_zero:
        print("Converting to huggingface Transformers format...")
        ckpt = torch.load(checkpoint_callback.best_model_path, map_location="cpu")
        finetuner.load_state_dict(ckpt["state_dict"])
        save_path = checkpoint_callback.best_model_path.replace('.ckpt', '')
        finetuner.model.save_pretrained(args.finetuned_ckpt)
        finetuner.tokenizer.save_pretrained(args.finetuned_ckpt)
        print(f"Model saved to {args.finetuned_ckpt}")


if __name__ == "__main__":
    args = parse_args()
    train(args)
