from __future__ import annotations
from typing import List, Optional
from dataclasses import dataclass
import numpy as np
import numba
from functools import cached_property


@dataclass
class Index:
    idstr: str
    kary: Optional[int] = None
    sep: str = "-"

    @cached_property
    def _idseq(self) -> List[int]:
        return list(map(int, self.idstr.split(self.sep)))
    
    def to_string(self) -> str:
        return self.idstr

    @property
    def positioned_index(self):
        assert self.kary is not None
        return [i * self.kary + idx + 2 for i, idx in enumerate(self._idseq)] + [1]
    
    @property
    def softmax_index(self):
        assert self.kary is not None
        idseq = np.array(self._idseq)
        return _generate_softmax_index(idseq)

    @classmethod
    def from_positioned_index(cls, positioned_index: List[int], kary: int, sep='-') -> Index:
        """
        Param:
            seqs: 2d ndarray to be decoded
        Return:
            doc_id string, List[str]
        """
        try:
            eos_idx = positioned_index.index(1)
            idseq = positioned_index[1:eos_idx]
        except:
            print("no eos token found")

        try:
            offset = np.arange(len(idseq)) * kary + 2
            idseq = np.array(idseq) - offset
        except Exception as e:
            print(f"offset: {offset}")
            print(f"idseq: {idseq}")
            raise e
        # assert np.all(res >= 0)
        idstr = sep.join(map(str, idseq.tolist()))
        return cls(idstr=idstr, kary=kary, sep=sep)


@numba.jit(nopython=True)
def _generate_softmax_index(idseq):
    softmax_index = 0
    for idx, num in enumerate(idseq[::-1]):
        softmax_index += num * (10**idx)
    return softmax_index
