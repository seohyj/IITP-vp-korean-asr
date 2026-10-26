"""Trie 기반 키워드 부스팅 (decoder-단 contextual biasing, 재학습 없음).

Whisper generate 의 logits_processor 로 꽂아, 키워드 토큰열의 prefix 를 따라가는
후보 토큰의 logit 에 +lambda 를 더한다. 가중치 업데이트 없이 추론 시 특정 어휘
(보이스피싱 수사/금융 고유명사 등) 인식 확률을 즉시 끌어올린다.

설계 요점:
  - beam search 는 스텝마다 beam 을 재정렬/가지치기 하므로, beam 인덱스에 상태를
    저장하면 깨진다. → 매 스텝 input_ids 의 suffix 로 trie 위치를 '재도출'하는
    stateless 방식 (Aho-Corasick 유사, 가장 긴 suffix 매칭).
  - 한국어는 조사가 붙으므로(검거→검거를/검거가) 단어 '종료'를 강제하지 않고
    prefix 진행 토큰만 boost → 조사 자연 생성 유지.
  - 토큰화는 공백-prefix 변이를 둘 다 등록: "검거" 와 " 검거"(문중 등장형)는
    Whisper BPE 에서 다른 토큰열 → 양쪽 경로 모두 trie 에 넣어야 실제로 걸린다.
  - root 자식(키워드 첫 토큰)은 어느 위치에서도 boost(=키워드 시작 허용).
    이 때문에 과부스팅(오삽입) 위험이 있어 lambda 와 anti-keyword 평가로 통제한다.

사용:
  proc = KeywordBoostProcessor.from_file(tokenizer, "keywords_vp.txt", lam=2.0)
  model.generate(..., logits_processor=[proc])
"""
from __future__ import annotations

import torch

try:
    from transformers import LogitsProcessor
except Exception:  # 타입만 필요 — 최소 shim
    class LogitsProcessor:  # type: ignore
        pass


class _Node:
    __slots__ = ("children", "is_end", "weight")

    def __init__(self, weight: float = 1.0):
        self.children: dict[int, "_Node"] = {}
        self.is_end = False
        self.weight = weight


class KeywordBoostProcessor(LogitsProcessor):
    def __init__(self, keyword_token_seqs, lam: float = 2.0,
                 lam_start: float | None = None,
                 boost_start_anywhere: bool = True):
        # dict 인지 list 인지 호환성 처리
        if isinstance(keyword_token_seqs, list):
            seqs_weights = {seq: 1.0 for seq in keyword_token_seqs}
        else:
            seqs_weights = keyword_token_seqs
            
        self.lam = float(lam)
        self.lam_start = float(lam) if lam_start is None else float(lam_start)
        self.boost_start_anywhere = boost_start_anywhere
        self.root = _Node(0.0)
        self.maxlen = 0
        for seq, weight in seqs_weights.items():
            if not seq:
                continue
            node = self.root
            for tid in seq:
                nxt = node.children.get(int(tid))
                if nxt is None:
                    nxt = _Node(weight)
                    node.children[int(tid)] = nxt
                else:
                    nxt.weight = max(nxt.weight, weight)
                node = nxt
            node.is_end = True
            self.maxlen = max(self.maxlen, len(seq))
        self.root_children = list(self.root.children.keys())
        self.n_keywords = len(seqs_weights)

    # ---------- 구성 ----------
    @classmethod
    def from_file(cls, tokenizer, path: str, lam: float = 2.0,
                  lam_start: float | None = None, variants: bool = True):
        """키워드 파일(한 줄 1개, 단어 \\t 가중치) → 토큰열 trie.
        가중치가 생략되면 1.0으로 간주.
        """
        seqs_weights = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                w = line.strip()
                if not w or w.startswith("#"):
                    continue
                parts = w.split("\t")
                word = parts[0]
                weight = float(parts[1]) if len(parts) > 1 else 1.0
                
                forms = [word, " " + word] if variants else [word]
                for form in forms:
                    ids = tokenizer.encode(form, add_special_tokens=False)
                    if ids:
                        seq_tuple = tuple(ids)
                        seqs_weights[seq_tuple] = max(seqs_weights.get(seq_tuple, 0.0), weight)
        return cls(seqs_weights, lam=lam, lam_start=lam_start)

    # ---------- 매칭 ----------
    def _match_node(self, tokens: list[int]) -> _Node:
        """tokens 의 가장 긴 suffix 가 root 에서 걸어지는 지점(부분 매칭 노드) 반환."""
        if self.maxlen == 0:
            return self.root
        maxl = min(self.maxlen, len(tokens))
        for L in range(maxl, 0, -1):
            node = self.root
            ok = True
            for tid in tokens[-L:]:
                nxt = node.children.get(tid)
                if nxt is None:
                    ok = False
                    break
                node = nxt
            if ok:
                return node
        return self.root

    # ---------- LogitsProcessor ----------
    def __call__(self, input_ids, scores):
        if self.maxlen == 0 or (self.lam == 0.0 and self.lam_start == 0.0):
            return scores
        root_set = set(self.root_children)
        for b in range(input_ids.shape[0]):
            toks = input_ids[b].tolist()
            node = self._match_node(toks)
            # 연속(continuation): 이미 키워드 prefix 를 따라가는 중인 다음 토큰.
            #   root 노드면 부분매칭 없음 → 연속 집합은 비어있음.
            cont = set(node.children.keys()) if node is not self.root else set()
            # 새 시작(start): 어느 위치에서든 키워드 첫 토큰. cont 와 겹치면 cont 우선.
            start = (root_set - cont) if self.boost_start_anywhere else set()
            if self.lam != 0.0 and cont:
                tids = sorted(cont)
                idx = torch.as_tensor(tids, device=scores.device, dtype=torch.long)
                w_vals = [node.children[tid].weight for tid in tids]
                w_tensor = torch.tensor(w_vals, device=scores.device, dtype=scores.dtype)
                scores[b, idx] += self.lam * w_tensor
            if self.lam_start != 0.0 and start:
                tids = sorted(start)
                idx = torch.as_tensor(tids, device=scores.device, dtype=torch.long)
                w_vals = [self.root.children[tid].weight for tid in tids]
                w_tensor = torch.tensor(w_vals, device=scores.device, dtype=scores.dtype)
                scores[b, idx] += self.lam_start * w_tensor
        return scores
