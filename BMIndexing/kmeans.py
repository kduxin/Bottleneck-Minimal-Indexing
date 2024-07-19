import os
from time import ctime
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import numpy as np
import pandas as pd
import pickle
import argparse
from tqdm import trange
import h5py
import cupy
import cuml

from NCIRetriever.io import IndexedEmbeddings, StringIndexing


def main(args):

    with h5py.File(args.embedding_path, "r") as f:
        X = f["embs"][:]
        ids = f["ids"][:]

    embeddings = IndexedEmbeddings.from_h5(args.embedding_path)
    X, ids = embeddings.embs, embeddings.ids
    print(X.shape)

    kmeans = cuml.KMeans(
        n_clusters=args.k,
        max_iter=300,
        n_init=args.n_init,
        init="scalable-k-means++",
        random_state=args.seed,
        tol=args.tol,
    )

    def classify_recursion(positions, level):
        n = len(positions)
        if n <= args.c:
            if n == 1:
                return
            indexing[positions, level] = cupy.arange(n)
            return

        preds = kmeans.fit_predict(X[positions])

        for i in range(args.k):
            subcluster_positions = positions[cupy.where(preds == i)[0]]
            indexing[subcluster_positions, level] = i
            classify_recursion(subcluster_positions, level + 1)

    max_index_length = 20
    X = cupy.array(X, dtype=cupy.float32)
    indexing = cupy.full(
        (X.shape[0], max_index_length), fill_value=-1, dtype=cupy.int32
    )

    print(ctime(), "Start First Clustering")
    preds = kmeans.fit_predict(X)
    print(preds.shape)  # int 0-9 for each vector
    print(kmeans.n_iter_)

    level = 0
    indexing[:, level] = preds

    print(ctime(), "Start Recursively Clustering...")
    for i in trange(args.k):
        positions = cupy.where(preds == i)[0]
        classify_recursion(positions, level + 1)

    docid2index = {}
    for i, index in enumerate(indexing.tolist()):
        end = index.index(-1)
        docid2index[ids[i]] = index[:end]
    indexing = StringIndexing(docid2index)

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    indexing.to_tsv(args.output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--v_dim", type=int, default=768)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--c", type=int, default=100)
    parser.add_argument("--n_init", type=int, default=100)
    parser.add_argument("--tol", type=float, default=1e-6)
    args = parser.parse_args()

    main(args)
