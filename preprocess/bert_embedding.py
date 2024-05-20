import os
import shutil
import argparse
import multiprocessing
import tqdm
import pandas as pd
import numpy as np
import torch
import transformers
from NCI.types import IndexedEmbeddings

torch.backends.cuda.matmul.allow_tf32 = True

cache = argparse.Namespace()


def main(args):
    df = pd.read_csv(
        args.docs_path,
        sep="\t",
        usecols=["docid", args.text_col],
        dtype={"docid": int, args.text_col: str},
    )

    device_que = multiprocessing.Queue()
    for i in range(args.n_gpus):
        device_que.put(i)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)

    batch_size = 20
    with multiprocessing.Pool(
        args.n_gpus, initializer=init, initargs=(args, device_que)
    ) as pool:
        ids, docs = df["docid"].tolist(), df[args.text_col].tolist()

        docids = []
        embeddings = []

        for docids_chunk, embeddings_chunk in zip(
            chunker(ids, batch_size),
            pool.imap(encode, chunker(tqdm.tqdm(docs), batch_size)),
        ):
            for docid, embedding in zip(docids_chunk, embeddings_chunk):
                if docid is None:
                    continue
                docids.append(docid)
                embeddings.append(embedding)

        embeddings = np.stack(embeddings)

    indexed_embeddings = IndexedEmbeddings(docids, embeddings)
    indexed_embeddings.to_h5(args.output_path)


def chunker(iterable, n, *, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks."
    from itertools import zip_longest

    # grouper('ABCDEFG', 3, fillvalue='x') → ABC DEF Gxx
    iterators = [iter(iterable)] * n
    return zip_longest(*iterators, fillvalue=fillvalue)


def init(args, device_que):
    print("Initializing ...")

    cache.args = args
    device = device_que.get()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    cache.tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_path)
    cache.model = transformers.AutoModel.from_pretrained(args.model_path).to(
        "cuda"
    )
    cache.model.eval()

    print("Initialization finished")


@torch.inference_mode()
def encode(docs):
    args = cache.args

    # docs = list(filter(None, docs))
    docs = [doc for doc in docs if doc is not None]

    inputs = cache.tokenizer(
        docs,
        max_length=args.max_len,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).to("cuda")
    outputs = cache.model(**inputs, return_dict=True)
    embeddings = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()

    return embeddings


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        """Convert documents to BERT embeddings.
        The `docs` file should be a TSV that has two columns.
        The first column should contain ids.
        The Second column should contain the text content.
        """
    )

    parser.add_argument("--docs_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--model_path", type=str, default="bert-base-uncased")
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--n_gpus", type=int)
    parser.add_argument("--text_col", type=str, default="doc")

    args = parser.parse_args()
    main(args)
