"""viz_token_score.py

BPE-token (tokens, token_scores) -> RDKit atom index mapping + visualization.

This version is BPE-safe:
- BPE tokens are arbitrary substrings (e.g. 'CCC', 'CC', 'C'), so you cannot consume atoms with a cursor.
- Instead, align tokens to the original SMILES by character spans, then map each token span to the atom spans it overlaps.

IMPORTANT:
- `smiles` must be EXACTLY the original string used for tokenization (do not canonicalize).

Deps: rdkit, numpy, matplotlib
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Tuple, Literal

import numpy as np
from matplotlib import cm
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D


Agg = Literal["max", "sum", "mean"]


@dataclass
class TokenAtomMapping:
    token_index: int
    token: str
    score: float
    span: Tuple[int, int]  # [start, end) in SMILES
    atom_indices: List[int]


# -------------------------
# 1) Token <-> SMILES alignment (char spans)
# -------------------------


def _strip_bpe_markers(tok: str) -> str:
    """Remove common BPE markers using unicode escapes (ASCII-only source code).

    - SentencePiece marker: U+2581 (\u2581)
    - GPT-2 BPE marker: U+0120 (\u0120)
    """
    if not tok:
        return tok
    if tok.startswith("\u2581"):
        return tok[1:]
    if tok.startswith("\u0120"):
        return tok[1:]
    return tok


def align_tokens_to_smiles(tokens: List[str], smiles: str) -> List[Tuple[int, int]]:
    """Compute each token's [start,end) char span in the SMILES string.

    Requirement: concatenating (after stripping BPE markers) must equal smiles.
    """
    spans: List[Tuple[int, int]] = []
    pos = 0

    for i, raw_tok in enumerate(tokens):
        tok = _strip_bpe_markers(raw_tok)
        if tok == "":
            spans.append((pos, pos))
            continue

        end = pos + len(tok)
        if end > len(smiles) or smiles[pos:end] != tok:
            context = smiles[max(0, pos - 20) : min(len(smiles), pos + 40)]
            raise ValueError(
                "Token/SMILES misalignment at token_index={} token={!r}.\n"
                "Expected SMILES[{}:{}] == token, but got SMILES segment={!r}.\n"
                "SMILES context around pos {}: {!r}\n"
                "Tip: make sure `smiles` is exactly the original string used for tokenization (not canonicalized).".format(
                    i, raw_tok, pos, end, smiles[pos:end], pos, context
                )
            )

        spans.append((pos, end))
        pos = end

    if pos != len(smiles):
        raise ValueError(
            f"Token concatenation length {pos} != SMILES length {len(smiles)}. "
            "Check tokenizer output and SMILES input."
        )

    return spans


# -------------------------
# 2) SMILES -> atom spans (lexical scan)
# -------------------------


def smiles_atom_spans(smiles: str) -> List[Tuple[int, int, int]]:
    """Return a list of (atom_index, start, end) in the input SMILES.

    Lexical rules covered:
    - bracket atoms: [nH], [NH+], [O-], [C@H]...
    - organic subset / common: B C N O P S F I
    - common 2-char: Cl Br
    - aromatic: b c n o p s
    - wildcard: *

    It ignores bonds, ring digits, branches, stereochem symbols, etc.

    Assumption: RDKit atom indices follow the order atoms appear in the SMILES.
    """
    spans: List[Tuple[int, int, int]] = []
    i = 0
    atom_idx = 0

    while i < len(smiles):
        ch = smiles[i]

        if ch == "[":
            j = i + 1
            while j < len(smiles) and smiles[j] != "]":
                j += 1
            if j < len(smiles) and smiles[j] == "]":
                spans.append((atom_idx, i, j + 1))
                atom_idx += 1
                i = j + 1
                continue

        if ch == "*":
            spans.append((atom_idx, i, i + 1))
            atom_idx += 1
            i += 1
            continue

        if i + 1 < len(smiles) and smiles[i : i + 2] in ("Cl", "Br"):
            spans.append((atom_idx, i, i + 2))
            atom_idx += 1
            i += 2
            continue

        if ch in ("b", "c", "n", "o", "p", "s"):
            spans.append((atom_idx, i, i + 1))
            atom_idx += 1
            i += 1
            continue

        if ch in ("B", "C", "N", "O", "P", "S", "F", "I"):
            spans.append((atom_idx, i, i + 1))
            atom_idx += 1
            i += 1
            continue

        i += 1

    return spans


# -------------------------
# 3) token spans -> atom indices (overlap)
# -------------------------


def _span_overlaps(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def token_span_to_atom_indices(
    token_span: Tuple[int, int],
    atom_spans: List[Tuple[int, int, int]],
    *,
    start_atom_ptr: int = 0,
) -> Tuple[List[int], int]:
    ts, te = token_span
    atom_indices: List[int] = []

    p = start_atom_ptr
    while p < len(atom_spans) and atom_spans[p][2] <= ts:
        p += 1

    q = p
    while q < len(atom_spans):
        ai, astart, aend = atom_spans[q]
        if astart >= te:
            break
        if _span_overlaps((ts, te), (astart, aend)):
            atom_indices.append(ai)
        q += 1

    return atom_indices, p


# -------------------------
# Public API
# -------------------------


def tokens_scores_to_atom_scores(
    smiles: str,
    tokens: List[str],
    token_scores: List[float],
    *,
    agg: Agg = "max",
    allow_mismatch: bool = False,
) -> Tuple[Chem.Mol, np.ndarray, List[TokenAtomMapping]]:
    """Map (tokens, token_scores) onto RDKit atom indices (BPE-safe)."""
    if len(tokens) != len(token_scores):
        raise ValueError("tokens and token_scores must have the same length")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")

    tok_spans = align_tokens_to_smiles(tokens, smiles)
    atom_sp = smiles_atom_spans(smiles)

    if not allow_mismatch and len(atom_sp) != mol.GetNumAtoms():
        raise ValueError(
            f"Atom span count ({len(atom_sp)}) != RDKit mol atom count ({mol.GetNumAtoms()}). "
            "If your SMILES contains elements beyond the scanner (e.g., Si, Na, ...), extend smiles_atom_spans() "
            "or set allow_mismatch=True."
        )

    atom_score_lists: List[List[float]] = [[] for _ in range(mol.GetNumAtoms())]
    mappings: List[TokenAtomMapping] = []

    atom_ptr = 0
    for i, (tok, sc, sp) in enumerate(zip(tokens, token_scores, tok_spans)):
        atom_indices, atom_ptr = token_span_to_atom_indices(sp, atom_sp, start_atom_ptr=atom_ptr)
        for ai in atom_indices:
            if 0 <= ai < len(atom_score_lists):
                atom_score_lists[ai].append(float(sc))
        mappings.append(TokenAtomMapping(i, tok, float(sc), sp, atom_indices))

    atom_scores = np.zeros(mol.GetNumAtoms(), dtype=float)
    for ai, lst in enumerate(atom_score_lists):
        if not lst:
            atom_scores[ai] = 0.0
        elif agg == "max":
            atom_scores[ai] = float(np.max(lst))
        elif agg == "sum":
            atom_scores[ai] = float(np.sum(lst))
        elif agg == "mean":
            atom_scores[ai] = float(np.mean(lst))
        else:
            raise ValueError(f"Unknown agg={agg}")

    return mol, atom_scores, mappings


# -------------------------
# Visualization
# -------------------------


def _normalize_for_highlight(atom_scores: np.ndarray, highlight_atoms: List[int]) -> np.ndarray:
    out = np.zeros_like(atom_scores)
    if not highlight_atoms:
        return out

    vals = atom_scores[highlight_atoms]
    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    if abs(vmax - vmin) < 1e-12:
        out[highlight_atoms] = 1.0
        return out

    out[highlight_atoms] = (vals - vmin) / (vmax - vmin)
    return out


def atom_scores_to_colors(
    atom_scores: np.ndarray,
    *,
    cmap_name: str = "coolwarm",
    eps: float = 1e-12,
) -> Tuple[Dict[int, Tuple[float, float, float]], List[int]]:
    highlight_atoms = [int(i) for i, s in enumerate(atom_scores) if float(s) > eps]
    norm = _normalize_for_highlight(atom_scores, highlight_atoms)
    cmap = cm.get_cmap(cmap_name)

    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    for ai in highlight_atoms:
        r, g, b, _a = cmap(float(norm[ai]))
        atom_colors[ai] = (float(r), float(g), float(b))

    return atom_colors, highlight_atoms


def draw_molecule_attention(
    smiles: str,
    atom_scores: np.ndarray,
    *,
    size: Tuple[int, int] = (450, 350),
    cmap: str = "coolwarm",
    kekulize: bool = True,
) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    if kekulize:
        try:
            Chem.Kekulize(mol)
        except Exception:
            pass

    atom_colors, highlight_atoms = atom_scores_to_colors(atom_scores, cmap_name=cmap)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    rdMolDraw2D.PrepareAndDrawMolecule(
        drawer,
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")


def save_svg(svg: str, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


if __name__ == "__main__":
    # Smoke test: char-level tokens always align
    smiles = "C#CC1=CC=CC(N/C2=N/C=N\\C3=CC(OCCOC)=C(OCCOC)C=C23)=C1"
    tokens = list(smiles)
    token_scores = [0.1] * len(tokens)

    mol, atom_scores, mappings = tokens_scores_to_atom_scores(smiles, tokens, token_scores, agg="max")
    svg = draw_molecule_attention(smiles, atom_scores)
    save_svg(svg, "viz_token_score.svg")
    print("saved viz_token_score.svg")
