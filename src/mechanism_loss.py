"""
Condition-difference regularizer (W2 scaffold).
Corollary discharge: HC show attenuated N100 for self-generated tones;
suppression reduced in SCZ. M5: requires PAIRED self/passive embeddings
(trial pairing logic is W2 TODO — counts typically differ per condition).

AUDIT 2026-11-10 (B13, design concern - deliberately NOT silently "fixed")
--------------------------------------------------------------------------
This loss pushes self/passive embeddings apart toward `margin` for every
sample identically: it is label-blind, so it encodes "self and passive must
differ" but not "differ less in HC, more in SCZ". As written it is a
contrastive scaffold, not a mechanism prior; enabling it as-is would let the
paper claim a mechanism constraint it does not implement. Make it
label-conditioned (or target a per-subject suppression ratio) before use.
"""

from __future__ import annotations
import torch
import torch.nn.functional as F
 
 
def condition_difference_loss(emb_self, emb_passive, margin=1.0):
    if emb_self.shape != emb_passive.shape:
        raise ValueError(
            "expects PAIRED embeddings with identical shapes, got "
            f"{tuple(emb_self.shape)} vs {tuple(emb_passive.shape)}")
    d = F.pairwise_distance(emb_self, emb_passive)
    return F.relu(margin - d).mean()
