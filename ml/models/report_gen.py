"""Experimental report generator: our CNN encoder + a small Transformer decoder trained from scratch.

Encoder: DenseNet-121 trunk (initialised from OUR chest classifier checkpoint when available).
Decoder: 4-layer Transformer, d_model 384 (~10M params), BPE vocabulary trained on IU X-ray reports.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class ReportGenerator(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 384,
        layers: int = 4,
        heads: int = 6,
        ffn: int = 1536,
        max_len: int = 128,
        dropout: float = 0.1,
        encoder: str = "densenet121",
    ) -> None:
        super().__init__()
        import timm

        self.encoder = timm.create_model(encoder, pretrained=False, num_classes=0, global_pool="")
        self.proj = nn.Linear(self.encoder.num_features, d_model)
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.img_pos = nn.Parameter(torch.zeros(1, 64, d_model))
        layer = nn.TransformerDecoderLayer(
            d_model, heads, ffn, dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size, bias=False)
        self.out.weight = self.tok.weight  # weight tying
        self.d_model, self.max_len = d_model, max_len
        nn.init.normal_(self.pos, std=0.02)
        nn.init.normal_(self.img_pos, std=0.02)

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        f = self.encoder(images)  # (B, C, h, w)
        f = f.flatten(2).transpose(1, 2)  # (B, hw, C)
        mem = self.proj(f)
        return mem + self.img_pos[:, : mem.shape[1]]

    def decode(self, memory: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        t = tokens.shape[1]
        x = self.tok(tokens) * math.sqrt(self.d_model) + self.pos[:, :t]
        mask = torch.triu(torch.full((t, t), float("-inf"), device=tokens.device), 1)
        h = self.decoder(x, memory, tgt_mask=mask)
        return self.out(self.norm(h))

    def forward(self, images: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(images), tokens)

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        bos: int,
        eos: int,
        max_new: int = 96,
        beam: int = 3,
        no_repeat_ngram: int = 3,
        min_len: int = 24,
        alpha: float = 1.0,
    ) -> list[list[int]]:
        """Beam search with n-gram blocking (batch of 1 for beams > 1)."""
        mem = self.encode(images)
        results = []
        for b in range(images.shape[0]):
            m = mem[b : b + 1]
            beams: list[tuple[list[int], float]] = [([bos], 0.0)]
            for _ in range(max_new):
                cand = []
                for seq, score in beams:
                    if seq[-1] == eos:
                        cand.append((seq, score))
                        continue
                    logits = self.decode(m, torch.tensor([seq], device=images.device))[0, -1]
                    logp = torch.log_softmax(logits, -1)
                    for ban in _banned(seq, no_repeat_ngram):
                        logp[ban] = -1e9
                    if len(seq) <= min_len:
                        logp[eos] = -1e9
                    top = torch.topk(logp, beam)
                    for lp, idx in zip(top.values.tolist(), top.indices.tolist()):
                        cand.append((seq + [idx], score + lp))
                beams = sorted(cand, key=lambda s: s[1] / len(s[0]) ** alpha, reverse=True)[:beam]
                if all(s[-1] == eos for s, _ in beams):
                    break
            best = beams[0][0]
            results.append([t for t in best[1:] if t != eos])
        return results


def _banned(seq: list[int], n: int) -> list[int]:
    if len(seq) < n:
        return []
    prefix = tuple(seq[-(n - 1) :])
    return [seq[i + n - 1] for i in range(len(seq) - n + 1) if tuple(seq[i : i + n - 1]) == prefix]
