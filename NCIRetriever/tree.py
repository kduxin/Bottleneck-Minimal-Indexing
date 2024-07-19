from typing import List, Set
import os
import pickle
import tqdm
import xxhash
from .data import DocidIndexCompatibility
from .index import Index



class Node(object):
    def __init__(self, token_id) -> None:
        self.token_id = token_id
        self.children = {}

    def __str__(self, level=0):
        ret = "\t" * level + repr(self.token_id) + "\n"
        for child in self.children.values():
            ret += child.__str__(level + 1)
        return ret

    def __repr__(self):
        return "<tree node representation>"


class IndexTree(object):
    root: Node

    def __init__(self, args) -> None:
        self.args = args
        self.root = Node(0)
        self.build(args)

    def build(self, args) -> Node:
        compat = DocidIndexCompatibility.from_tsv(
            f"{args.data_root}/{args.data_subdir}/docid2index.tsv"
        )
        idstrs: List[str] = compat.indexes

        hsh = xxhash.xxh128_hexdigest("|".join(idstrs))
        cache_path = f"{args.data_root}/{args.data_subdir}/cache/{hsh}.pkl"
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)

        if os.path.isfile(cache_path):
            with open(cache_path, "rb") as input_file:
                self.root = pickle.load(input_file)
            print(f"tree loaded from {cache_path}")

        else:
            for idstr in tqdm.tqdm(idstrs):
                index = Index(idstr, args.kary)
                self.add(index.positioned_index)

            with open(cache_path, "wb") as output_file:
                pickle.dump(self.root, output_file)
            print(f"tree saved to {cache_path}")

    def add(self, index: List[int]) -> None:
        """
        seq is List[Int] representing id, without leading pad_token(hardcoded 0), with trailing eos_token(hardcoded 1) and (possible) pad_token, every int is token index
        e.g: [ 9, 14, 27, 38, 47, 58, 62,  1,  0,  0,  0,  0,  0,  0,  0]
        """
        cur = self.root
        for idx in index:
            if idx == 0:  # reach pad_token
                return
            if idx not in cur.children:
                cur.children[idx] = Node(idx)
            cur = cur.children[idx]

    def encode_index(self, idstr: str) -> List[int]:
        """
        Param:
            seq: doc_id string to be encoded, like "23456"
        Return:
            List[Int]: encoded tokens
        """
        args = self.args

        target_id_int = []
        for i, c in enumerate(idstr.split("-")):
            if args.position:
                cur_token = i * args.kary + int(c) + 2
            else:
                cur_token = int(c) + 2
            target_id_int.append(cur_token)
        return target_id_int + [1]  # append eos_token
