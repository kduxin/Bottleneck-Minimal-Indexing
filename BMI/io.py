from __future__ import annotations
from typing import Tuple, List, Mapping, Set, Hashable
from dataclasses import dataclass, field
from collections import defaultdict
import tqdm
import h5py
import numpy as np
import pandas as pd
from numpy import ndarray

from .index import Index


@dataclass
class IndexedEmbeddings:
    ids: ndarray
    embs: ndarray

    def __post_init__(self):
        assert len(self.ids) == len(self.embs)

    @classmethod
    def from_tsv(cls, path):
        ids = []
        embs = []
        with open(path, "rt") as f:
            for line in tqdm.tqdm(f):
                idx, embstr = line.strip().split("\t")
                ids.append(idx)
                emb = np.fromstring(embstr, sep="|")
                embs.append(emb)

        return cls(ids=np.array(ids), embs=np.stack(embs))

    def to_tsv(self, path):
        format = "{:g}"
        with open(path, "wt") as f:
            for idx, emb in zip(tqdm.tqdm(self.ids), self.embs):
                embstr = "|".join(map(format.format, emb.tolist()))
                f.write(f"{idx}\t{embstr}\n")

    @classmethod
    def from_h5(cls, path):
        with h5py.File(path, "r") as f:
            ids = f["ids"][:]
            embs = f["embs"][:]
        if ids.dtype == 'O':
            ids = ids.astype('str')
        return cls(ids=ids, embs=embs)

    def to_h5(self, path):
        with h5py.File(path, "w") as f:
            f.create_dataset("ids", data=self.ids)
            f.create_dataset(
                "embs", data=self.embs.astype(np.float32), dtype=np.float32
            )

    def to_pandas(self, columns=["id", "emb"]):
        return pd.DataFrame({columns[0]: self.ids, columns[1]: list(self.embs)})

    def __getitem__(self, x):
        return self.ids[x], self.embs[x]

    def __len__(self, x):
        return len(self.ids)


@dataclass
class StringIndexing:
    docid2index: Mapping[Hashable, List[int]]
    index2docid: Mapping[str, Hashable]

    def __init__(self, docid2index: Mapping[Hashable, List[int]]):
        self.docid2index = docid2index
        self.index2docid = {
            intarray_to_string(val): key for key, val in docid2index.items()
        }

    @classmethod
    def from_tsv(cls, path):
        docid2index = {}
        with open(path, "rt") as f:
            lines = f.readlines()
            for line in lines[1:]:
                docid, ids = line.strip().split("\t")
                docid2index[docid] = tuple(map(int, ids.split("-")))
        return cls(docid2index)

    def to_tsv(self, path):
        with open(path, "wt") as f:
            f.write(f"docid\tindex\n")
            for idx, ids in self.docid2index.items():
                f.write(f"{idx}\t" + "-".join(map(str, ids)) + "\n")

    def to_pandas(self):
        return pd.DataFrame(self.docid2index.items(), columns=["docid", "index"])

    def __getitem__(self, docid: Hashable):
        return self.docid2index[docid]

    def get_docid(self, index: str | Tuple[int] | List[int]):
        assert isinstance(index, (str, list, tuple))
        if isinstance(index, (list, tuple)):
            index = intarray_to_string(index)
        return self.index2docid[index]

    def get_index(self, docid: int):
        return self.docid2index[docid]

    def __len__(self):
        return len(self.docid2index)


def intarray_to_string(index):
    return "-".join(map(str, index))


@dataclass
class DocumentRetrievalTrainingFile:
    queries: List[str]
    indexes: List[str]
    docids: List[Hashable]

    def __post_init__(self):
        assert len(self.queries) == len(self.indexes) == len(self.docids)

    @classmethod
    def from_tsv(cls, path):
        df = pd.read_csv(path, sep="\t", usecols=["query", "index", "docid"])
        return cls(
            queries=df["query"].tolist(),
            indexes=df["index"].tolist(),
            docids=df["docid"].tolist(),
        )

    def to_tsv(self, path):
        df = pd.DataFrame(
            {"query": self.queries, "index": self.indexes, "docid": self.docids}
        )
        df.to_csv(path, sep="\t", index=None)

    def to_pandas(self):
        return pd.DataFrame(
            {"query": self.queries, "index": self.indexes, "docid": self.docids}
        )


@dataclass
class DocumentRetrievalInferenceFile:
    queries: List[str]
    docids: List[Hashable]

    def __post_init__(self):
        assert len(self.queries) == len(self.docids)

    @classmethod
    def from_tsv(cls, query_docid_path):
        query_docid = pd.read_csv(
            query_docid_path, sep="\t", usecols=["query", "docid"],
            dtype={"query": str}
        )
        return cls(
            queries=query_docid["query"].tolist(),
            docids=query_docid["docid"].tolist(),
        )

    def to_tsv(self, query_docid_path):
        pd.DataFrame({"query": self.queries, "docid": self.docids}).to_csv(
            query_docid_path, sep="\t", index=None
        )

    def to_pandas(self):
        return pd.DataFrame({"query": self.queries, "docid": self.docids})


@dataclass
class DocidIndexCompatibility:
    indexes: List[str]
    docids: List[Hashable]
    index2docids: Mapping[str, Set[Hashable]] = field(init=False)
    docid2indexes: Mapping[Hashable, Set[str]] = field(init=False)

    def __post_init__(self):
        assert len(self.indexes) == len(self.docids)
        self._build_index_docid_maps()

    @classmethod
    def from_tsv(cls, path):
        df = pd.read_csv(
            path,
            sep="\t",
            usecols=["docid", "index"],
            dtype={"index": str},
        )
        return cls(indexes=df["index"].tolist(), docids=df["docid"].tolist())

    def to_tsv(self, path):
        df = pd.DataFrame({"docid": self.docids, "index": self.indexes})
        df.to_csv(path, sep="\t", header=None, index=None)

    def is_compatible(self, index: Index, docid: Hashable):
        assert isinstance(index, Index)
        idstr = index.to_string()
        return docid in self.index2docids[idstr] and idstr in self.docid2indexes[docid]

    def _build_index_docid_maps(self):
        self.index2docids = defaultdict(set)
        self.docid2indexes = defaultdict(set)
        for i, d in zip(self.indexes, self.docids):
            self.index2docids[i].add(d)
            self.docid2indexes[d].add(i)

