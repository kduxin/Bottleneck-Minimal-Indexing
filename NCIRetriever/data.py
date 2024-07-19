from __future__ import annotations
from typing import List, Mapping, Set
from dataclasses import dataclass
import pandas as pd
from torch.utils.data import Dataset

from .index import Index
from .io import (
    DocumentRetrievalTrainingFile,
    DocumentRetrievalInferenceFile,
    DocidIndexCompatibility,
)


class DocumentRetrievalTrainingDataset(Dataset):
    queries: List[str] = []
    indexes: List[Index] = []
    docids: List[int] = []

    def __init__(self, args):
        self.args = args
        self.queries, self.indexes, self.docids = self._load_data(args)

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        sample = DocumentRetrievalTrainingSample(
            query=self.queries[i], docid=self.docids[i], index=self.indexes[i]
        )
        return sample

    def _load_data(self, args):
        files = []
        if "realq" in args.query_type:
            files.append("realq_train.tsv")
        if "genq" in args.query_type:
            files.append("genq.tsv")
        if "ta" in args.query_type:
            files.append("title_abs.tsv")
        if "docseg" in args.query_type:
            files.append("docseg.tsv")

        # A sample should be a tuple of: (query, index, rank, softmax_index, neg_docid_list, aug_query_list)
        queries, indexes, docids = [], [], []
        for file in files:
            print(file)
            path = f"{args.data_root}/{args.data_subdir}/{file}"

            raw = DocumentRetrievalTrainingFile.from_tsv(path)
            queries.extend(raw.queries)
            indexes.extend([Index(idstr, kary=args.kary) for idstr in raw.indexes])
            docids.extend(raw.docids)
        return queries, indexes, docids


@dataclass
class DocumentRetrievalTrainingSample:
    query: str
    docid: int
    index: Index


class DocumentRetrievalInferenceDataset:
    queries: List[str]
    docids: List[int]
    compatibility: DocidIndexCompatibility

    def __init__(self, args):
        self.args = args

        data = DocumentRetrievalInferenceFile.from_tsv(
            f"{args.data_root}/{args.data_subdir}/realq_dev.tsv"
        )
        self.queries = data.queries
        self.docids = data.docids

        self.compatibility = DocidIndexCompatibility.from_tsv(
            f"{args.data_root}/{args.data_subdir}/docid2index.tsv"
        )

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, i):
        return DocumentRetrievalInferenceSample(
            query=self.queries[i], docid=self.docids[i]
        )


@dataclass
class DocumentRetrievalInferenceSample:
    query: str
    docid: int

