from typing import List
from time import ctime
import copy
import random
import numpy as np
import numba
from collections import defaultdict
import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .data import (
    DocumentRetrievalTrainingDataset,
    DocumentRetrievalTrainingSample,
    DocumentRetrievalInferenceDataset,
    DocumentRetrievalInferenceSample,
)
from .nci_transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup,
)
from .tree import IndexTree
from .index import Index


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class T5FineTuner(pl.LightningModule):
    tree: IndexTree

    def __init__(self, args):
        super(T5FineTuner, self).__init__()
        self.args = args
        self.save_hyperparameters(args)

        print(ctime(), "Building index tree...")
        self.tree = IndexTree(args)
        print(ctime(), "Index tree built.")

        # TODO: check if any index is longer than args.output_vocab_size - 2

        expand_scale = args.max_output_length
        self.decode_vocab_size = args.output_vocab_size * expand_scale + 2

        # Tokenizer initialization
        self.tokenizer = T5Tokenizer.from_pretrained(args.tokenizer_name_or_path)

        # Datasets initialization
        self.train_dataset = DocumentRetrievalTrainingDataset(self.args)
        self.val_dataset = DocumentRetrievalInferenceDataset(self.args)

        # Model initialization
        t5_config = T5Config(
            num_layers=args.num_layers,
            num_decoder_layers=args.num_decoder_layers,
            d_ff=args.d_ff,
            d_model=args.d_model,
            num_heads=args.num_heads,
            decoder_start_token_id=0,  # 1,
            output_past=True,
            d_kv=args.d_kv,
            dropout_rate=args.dropout_rate,
            decode_embedding=args.decode_embedding,
            decode_vocab_size=self.decode_vocab_size,
            output_vocab_size=args.output_vocab_size,
            tie_decode_embeddings=args.tie_decode_embeddings,
            tie_word_embeddings=args.tie_word_embeddings,
            Rdrop=args.Rdrop,
            Rdrop_only_decoder=args.Rdrop_only_decoder,
            Rdrop_loss=args.Rdrop_loss,
            adaptor_decode=args.adaptor_decode,
            adaptor_efficient=args.adaptor_efficient,
            adaptor_layer_num=args.adaptor_layer_num,
            train_batch_size=args.train_batch_size,
            eval_batch_size=args.eval_batch_size,
            max_output_length=args.max_output_length,
        )
        print(ctime(), t5_config)
        model = T5ForConditionalGeneration(t5_config)
        if args.load_pretrained_encoder:
            print(ctime(), "Loading pretrained model ...")
            pretrain_model = T5ForConditionalGeneration.from_pretrained(
                args.model_name_or_path
            )
            print(ctime(), "Modeled loaded.")
            pretrain_params = dict(pretrain_model.named_parameters())
            for name, param in model.named_parameters():
                if name.startswith(("shared.", "encoder.")):
                    with torch.no_grad():
                        param.copy_(pretrain_params[name])
            print(ctime(), "Parameters loaded.")
        self.model = model
        if args.verbose:
            print(ctime(), self.model)

        if self.args.freeze_embeds:
            self.freeze_embeds()
        if self.args.freeze_encoder:
            encoder = self.model.get_encoder()
            freeze_params(encoder)
            assert not any(p.requires_grad for p in encoder.parameters())

        self.training_outputs = defaultdict(list)
        self.validation_outputs = defaultdict(list)

    def freeze_embeds(self):
        """Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."""
        try:
            freeze_params(self.model.model.shared)
            for d in [self.model.model.encoder, self.model.model.decoder]:
                freeze_params(d.embed_positions)
                freeze_params(d.embed_tokens)
        except AttributeError:
            freeze_params(self.model.shared)
            for d in [self.model.encoder, self.model.decoder]:
                freeze_params(d.embed_tokens)

    def lmap(self, f, x):
        """list(map(f, x))"""
        return list(map(f, x))

    def is_logger(self):
        return self.trainer.global_rank <= 0

    def parse_score(self, result):
        return {k: round(v.mid.fmeasure * 100, 4) for k, v in result.items()}

    def forward(
        self,
        input_ids,
        aug_input_ids=None,
        encoder_outputs=None,
        attention_mask=None,
        aug_attention_mask=None,
        logit_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        query_embedding=None,
        prefix_emb=None,
        prefix_mask=None,
        decoder_index=-1,
        input_mask=None,
    ):
        input_mask = None
        if self.args.Rdrop > 0 and not self.args.Rdrop_only_decoder and self.training:
            if aug_input_ids is not None and self.training:
                input_ids = torch.cat([input_ids, aug_input_ids.clone()], dim=0)
                attention_mask = torch.cat([attention_mask, aug_attention_mask], dim=0)
            elif self.training:
                input_ids = torch.cat([input_ids, input_ids.clone()], dim=0)
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.clone()], dim=0
                )
            if decoder_attention_mask is not None:
                decoder_attention_mask = torch.cat(
                    [decoder_attention_mask, decoder_attention_mask], dim=0
                )
            if labels is not None:
                labels = torch.cat([labels, labels], dim=0)
            if decoder_input_ids is not None:
                decoder_input_ids = torch.cat(
                    [decoder_input_ids, decoder_input_ids], dim=0
                )

        out = self.model(
            input_ids,
            input_mask=input_mask,
            logit_mask=logit_mask,
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
            query_embedding=query_embedding,
            prefix_embedding=prefix_emb,
            prefix_mask=prefix_mask,
            return_dict=True,
            output_hidden_states=True,
            decoder_index=decoder_index,
        )
        return out

    def _softmax_generative_step(self, batch):
        assert self.args.softmax
        lm_labels = batch["target_ids"]
        lm_labels[lm_labels[:, :] == self.tokenizer.pad_token_id] = -100

        outputs = self.forward(
            input_ids=batch["source_ids"],
            attention_mask=batch["source_mask"],
            lm_labels=lm_labels,
            decoder_attention_mask=batch["target_mask"],
        )

        pred_index = torch.argmax(outputs[0], dim=1)
        return pred_index

    def ids_to_clean_text(self, generated_ids):
        gen_text = self.tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        return self.lmap(str.strip, gen_text)

    def training_step(self, batch: List[DocumentRetrievalTrainingSample], batch_idx):
        args = self.args

        source = self.tokenizer(
            [clean_text(sample.query) for sample in batch],
            return_tensors="pt",
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
        )

        aug_source = self.tokenizer(
            [clean_text(corrupt_query(sample.query)) for sample in batch],
            return_tensors="pt",
            max_length=args.max_input_length,
            padding="max_length",
            truncation=True,
        )

        def tokenize_target(indexes: List[Index]):
            labels = torch.zeros(len(indexes), args.max_output_length, dtype=torch.long)
            for i, index in enumerate(indexes):
                # 1 marks end of an index
                pindex = index.positioned_index
                labels[i, : len(pindex)] = torch.LongTensor(pindex)

            decoder_attention_mask = labels.clone()
            decoder_attention_mask[decoder_attention_mask != 0] = 1

            labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": labels, "attention_mask": decoder_attention_mask}

        target = tokenize_target([sample.index for sample in batch])

        outputs = self.forward(
            input_ids=source["input_ids"].to(self.device),
            aug_input_ids=aug_source["input_ids"].to(self.device),
            labels=target["input_ids"].to(self.device),
            attention_mask=source["attention_mask"].to(self.device),
            aug_attention_mask=aug_source["attention_mask"].to(self.device),
            decoder_attention_mask=target["attention_mask"].to(self.device),
            query_embedding=None,
            decoder_index=-1,
            encoder_outputs=None,
            prefix_emb=None,
            prefix_mask=None,
            input_mask=None,
        )

        self.log("train_loss", outputs.loss)
        res = {
            "loss": outputs.loss,
            "orig_loss": outputs.orig_loss if args.Rdrop > 0 else 0,
            "kl_loss": outputs.dist_loss if args.Rdrop > 0 else 0,
        }
        for key, val in detach_dict(res).items():
            self.training_outputs[key].append(val)
        return res

    def on_train_epoch_end(self):
        if not len(self.training_outputs["loss"]):
            return

        losses = self.training_outputs["loss"]

        avg_train_loss = torch.stack(losses).mean()
        self.log("avg_train_loss", avg_train_loss, sync_dist=True)

        self.training_outputs.clear()

    def validation_step(self, batch: List[DocumentRetrievalInferenceSample], batch_idx):
        args = self.args

        source = self.tokenizer(
            [clean_text(sample.query) for sample in batch],
            return_tensors="pt",
            max_length=self.args.max_input_length,
            padding="max_length",
            truncation=True,
        )

        decode_vocab_size = args.output_vocab_size * args.max_output_length + 2

        outs, scores = self.model.generate(
            source["input_ids"].to(self.device),
            attention_mask=source["attention_mask"].to(self.device),
            use_cache=False,
            # decoder_attention_mask=target_mask,
            decoder_attention_mask=None,
            max_length=args.max_output_length,
            num_beams=args.num_return_sequences,
            length_penalty=args.length_penalty,
            num_return_sequences=args.num_return_sequences,
            early_stopping=False,
            decode_embedding=args.decode_embedding,
            decode_vocab_size=decode_vocab_size,
            decode_tree=self.tree.root,
            decoder_index=-1,
            output_scores=True,
        )

        preds_all = [
            Index.from_positioned_index(idseq, kary=args.kary)
            for idseq in outs.cpu().numpy().tolist()
        ]

        preds_per_sample = [
            preds_all[i : i + args.num_return_sequences]
            for i in range(0, len(preds_all), args.num_return_sequences)
        ]

        compat = self.val_dataset.compatibility
        hits = []
        for sample, preds in zip(batch, preds_per_sample):
            # hit@1
            pred: Index = preds[0]
            if compat.is_compatible(pred, sample.docid):
                hits.append(1)
            else:
                hits.append(0)

        self.validation_outputs["hits"].extend(hits)

        # return {"inf_result_batch": inf_result_cache, "inf_result_batch_prob": scores}
        return hits

    def on_validation_epoch_end(self):
        if not len(self.validation_outputs["hits"]):
            return

        rec_at_1 = np.mean(self.validation_outputs["hits"])
        print(f"recall@1:{rec_at_1}")
        self.log("recall1", rec_at_1, sync_dist=True)

        self.validation_outputs.clear()

    def configure_optimizers(self):
        "Prepare optimizer and schedule (linear warmup and decay)"

        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay))
                    and (n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (not any(nd in n for nd in no_decay))
                    and (not n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": self.args.weight_decay,
                "lr": self.args.decoder_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay))
                    and (n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": 0.0,
                "lr": self.args.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (any(nd in n for nd in no_decay))
                    and (not n.startswith(("shared.", "encoder.")))
                ],
                "weight_decay": 0.0,
                "lr": self.args.decoder_learning_rate,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [
            {"scheduler": scheduler, "interval": "step", "frequency": 1}
        ]

    def get_tqdm_dict(self):
        tqdm_dict = {
            "loss": "{:.3f}".format(self.trainer.avg_loss),
            "lr": self.lr_scheduler.get_last_lr()[-1],
        }
        return tqdm_dict

    def train_dataloader(self):
        print("load training data and create training loader.")
        train_dataset = self.train_dataset
        if self.args.n_train > 0:
            train_dataset = [train_dataset[i] for i in range(self.args.n_train)]
        sampler = DistributedSampler(train_dataset)
        dataloader = DataLoader(
            train_dataset,
            collate_fn=lambda x: x,
            sampler=sampler,
            batch_size=self.args.train_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=4,
        )
        return dataloader

    def val_dataloader(self):
        print("load validation data and create validation loader.")
        val_dataset = self.val_dataset
        if self.args.n_val > 0:
            val_dataset = [val_dataset[i] for i in range(self.args.n_val)]
        sampler = DistributedSampler(val_dataset)
        dataloader = DataLoader(
            val_dataset,
            collate_fn=lambda x: x,
            sampler=sampler,
            batch_size=self.args.eval_batch_size,
            drop_last=True,
            shuffle=False,
            num_workers=4,
        )
        return dataloader


def detach_dict(d: dict):
    return {
        key: val.detach() if isinstance(val, torch.Tensor) else val
        for key, val in d.items()
    }


def clean_text(text):
    text = text.replace("\n", "")
    text = text.replace("``", "")
    text = text.replace('"', "")
    return text


@numba.jit(nopython=True)
def corrupt_query(query):
    if len(query) < 20 * 10:
        start_pos = np.random.randint(0, int(len(query) + 1 / 2))
        end_pos = np.random.randint(start_pos, len(query))
        span_length = max(end_pos - start_pos, 10 * 10)
        new_query = query[start_pos : start_pos + span_length]
    else:
        start_pos = np.random.randint(0, len(query) - 10 * 10)
        end_pos = np.random.randint(start_pos + 5 * 10, len(query))
        span_length = min(end_pos - start_pos, 20 * 10)
        new_query = query[start_pos : start_pos + span_length]
    return new_query


def freeze_params(model):
    for par in model.parameters():
        par.requires_grad = False
