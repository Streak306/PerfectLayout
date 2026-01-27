# -*- coding: utf-8 -*-
"""
Legends of Heropolis DX - Layout Planner (GUI helper core)

This module is intentionally small and dependency-free.
It supports:
- Spot bonus scoring (adjacency in the 8-neighborhood, including diagonals)
- Chunk rule: any footprint larger than 1x1 must fit entirely inside a single 4x4 chunk
- Blocked tiles (cells you mark red in the GUI)
- A heuristic optimizer that tries to maximize total % bonus
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Callable, Iterable
import json
import math
import random

# ---------------------------
# Data types
# ---------------------------

@dataclass(frozen=True)
class Placement:
    """One placed object instance on the map (tile coords)."""
    name: str
    x: int
    y: int
    w: int
    h: int

# ---------------------------
# Loading
# ---------------------------

def load_spots(path: str) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_pair_bonus(spots: List[dict]) -> Dict[Tuple[str, str], float]:
    """
    Precompute total bonus % for any unordered pair of item names.

    If a pair matches multiple spots, bonuses add.
    If a spot has A and B overlapping, adjacency between two items in that overlap still counts
    (because one can be "A" and the other "B"), but we count each unordered pair once.
    """
    pair_bonus: Dict[Tuple[str, str], float] = {}
    for s in spots:
        A = set(s.get("A", []))
        B = set(s.get("B", []))
        pct = float(s.get("pct", s.get("buff", {}).get("percent", 0.0)))
        # All pairs where one is in A and the other in B (unordered)
        for a in A:
            for b in B:
                if a == b:
                    # adjacency of two different instances of the same item is possible
                    # We'll represent as (a,a) key; scoring code handles instances.
                    key = (a, a)
                else:
                    key = tuple(sorted((a, b)))
                pair_bonus[key] = pair_bonus.get(key, 0.0) + pct
    return pair_bonus

# ---------------------------
# Geometry rules
# ---------------------------

def rects_overlap(a: Placement, b: Placement) -> bool:
    return not (a.x + a.w <= b.x or b.x + b.w <= a.x or a.y + a.h <= b.y or b.y + b.h <= a.y)

def rects_touch_8(a: Placement, b: Placement) -> bool:
    """
    Two rectangles "touch" if any tile of A is within Chebyshev distance 1 of any tile of B.
    Equivalent to bounding boxes being within 1 tile in both axes.
    """
    ax2 = a.x + a.w - 1
    ay2 = a.y + a.h - 1
    bx2 = b.x + b.w - 1
    by2 = b.y + b.h - 1
    if ax2 < b.x - 1: return False
    if bx2 < a.x - 1: return False
    if ay2 < b.y - 1: return False
    if by2 < a.y - 1: return False
    return True

def within_one_chunk(x: int, y: int, w: int, h: int, chunk_size: int = 4) -> bool:
    """
    Chunk rule:
    - 1x1 can be anywhere.
    - Any footprint larger than 1x1 must NOT cross a 4x4 chunk boundary.
      (It must fit entirely inside one chunk.)
    """
    if w == 1 and h == 1:
        return True
    # which chunk is top-left in?
    cx0 = x // chunk_size
    cy0 = y // chunk_size
    # which chunk is bottom-right in?
    cx1 = (x + w - 1) // chunk_size
    cy1 = (y + h - 1) // chunk_size
    return (cx0 == cx1) and (cy0 == cy1)

def rect_hits_blocked(x: int, y: int, w: int, h: int, blocked: set[Tuple[int,int]]) -> bool:
    for yy in range(y, y + h):
        for xx in range(x, x + w):
            if (xx, yy) in blocked:
                return True
    return False

def is_valid_position(
    placements: List[Placement],
    idx_ignore: Optional[int],
    x: int,
    y: int,
    w: int,
    h: int,
    map_w: int,
    map_h: int,
    blocked: set[Tuple[int,int]],
    chunk_size: int = 4
) -> bool:
    if x < 0 or y < 0 or x + w > map_w or y + h > map_h:
        return False
    if not within_one_chunk(x, y, w, h, chunk_size):
        return False
    if rect_hits_blocked(x, y, w, h, blocked):
        return False
    # overlap
    cand = Placement("?", x, y, w, h)
    for i, p in enumerate(placements):
        if idx_ignore is not None and i == idx_ignore:
            continue
        if rects_overlap(cand, p):
            return False
    return True

# ---------------------------
# Scoring
# ---------------------------

def score_layout(placements: List[Placement], pair_bonus: Dict[Tuple[str,str], float]) -> float:
    total = 0.0
    n = len(placements)
    for i in range(n):
        a = placements[i]
        for j in range(i+1, n):
            b = placements[j]
            if not rects_touch_8(a, b):
                continue
            key = (a.name, a.name) if a.name == b.name else tuple(sorted((a.name, b.name)))
            total += pair_bonus.get(key, 0.0)
    return total

# ---------------------------
# Free capacity helpers (reserved space constraints)
# ---------------------------

def occupied_tiles(placements: List[Placement]) -> set[Tuple[int,int]]:
    occ: set[Tuple[int,int]] = set()
    for p in placements:
        for yy in range(p.y, p.y + p.h):
            for xx in range(p.x, p.x + p.w):
                occ.add((xx, yy))
    return occ

def count_empty_clean_chunks(
    map_w: int,
    map_h: int,
    blocked: set[Tuple[int,int]],
    occ: set[Tuple[int,int]],
    chunk_size: int = 4,
) -> int:
    """Chunks 4x4 totalmente vazios e sem bloqueios."""
    count = 0
    for cy in range(0, map_h, chunk_size):
        if cy + chunk_size > map_h:
            break
        for cx in range(0, map_w, chunk_size):
            if cx + chunk_size > map_w:
                break
            has_block = False
            has_occ = False
            for yy in range(cy, cy + chunk_size):
                for xx in range(cx, cx + chunk_size):
                    if (xx, yy) in blocked:
                        has_block = True
                        break
                    if (xx, yy) in occ:
                        has_occ = True
                if has_block:
                    break
            if not has_block and not has_occ:
                count += 1
    return count

def count_free_2x2_slots(
    map_w: int,
    map_h: int,
    blocked: set[Tuple[int,int]],
    occ: set[Tuple[int,int]],
    chunk_size: int = 4,
) -> int:
    """Quantidade de posições 2x2 totalmente livres (sem cruzar chunk)."""
    free = 0
    for cy in range(0, map_h, chunk_size):
        if cy + chunk_size > map_h:
            break
        for cx in range(0, map_w, chunk_size):
            if cx + chunk_size > map_w:
                break
            for dy in range(0, chunk_size, 2):
                for dx in range(0, chunk_size, 2):
                    ok = True
                    for yy in range(cy + dy, cy + dy + 2):
                        for xx in range(cx + dx, cx + dx + 2):
                            if xx >= map_w or yy >= map_h or (xx, yy) in blocked or (xx, yy) in occ:
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        free += 1
    return free

def free_capacity_counts(
    placements: List[Placement],
    map_w: int,
    map_h: int,
    blocked: set[Tuple[int,int]],
    chunk_size: int = 4,
) -> Tuple[int,int,int]:
    """(chunks_4x4_limpos, slots_2x2_livres, tiles_1x1_livres)"""
    occ = occupied_tiles(placements)
    chunks = count_empty_clean_chunks(map_w, map_h, blocked, occ, chunk_size)
    slots2 = count_free_2x2_slots(map_w, map_h, blocked, occ, chunk_size)
    free1 = (map_w * map_h) - len(blocked) - len(occ)
    return chunks, slots2, free1

def free_capacity_ok(
    placements: List[Placement],
    map_w: int,
    map_h: int,
    blocked: set[Tuple[int,int]],
    reserve_empty_chunks: int,
    reserve_free_2x2: int,
    reserve_free_1x1: int,
    chunk_size: int = 4,
) -> bool:
    chunks, slots2, free1 = free_capacity_counts(placements, map_w, map_h, blocked, chunk_size)
    return (chunks >= reserve_empty_chunks) and (slots2 >= reserve_free_2x2) and (free1 >= reserve_free_1x1)

# ---------------------------
# Optimizer
# ---------------------------

def _candidate_positions(map_w: int, map_h: int, w: int, h: int, blocked: set[Tuple[int,int]], chunk_size: int) -> List[Tuple[int,int]]:
    pos = []
    for y in range(0, map_h - h + 1):
        for x in range(0, map_w - w + 1):
            if not within_one_chunk(x, y, w, h, chunk_size):
                continue
            if rect_hits_blocked(x, y, w, h, blocked):
                continue
            pos.append((x,y))
    return pos

def _initial_pack(instances: List[Tuple[str,int,int]], map_w: int, map_h: int, blocked: set[Tuple[int,int]], chunk_size: int, rng: random.Random, reserve_empty_chunks: int, reserve_free_2x2: int, reserve_free_1x1: int) -> Optional[List[Placement]]:
    """
    Greedy randomized packing, largest footprints first.
    instances: list of (name,w,h)
    """
    # Precompute candidates per footprint
    cand_cache: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    def get_cands(w,h):
        key=(w,h)
        if key not in cand_cache:
            cand_cache[key]=_candidate_positions(map_w,map_h,w,h,blocked,chunk_size)
        return cand_cache[key]

    order = sorted(instances, key=lambda t: (t[1]*t[2], max(t[1],t[2])), reverse=True)
    placements: List[Placement] = []
    for name,w,h in order:
        cands = get_cands(w,h)
        if not cands:
            return None
        # try a bunch of random candidates
        idxs = list(range(len(cands)))
        rng.shuffle(idxs)
        placed = False
        for k in idxs:
            x,y = cands[k]
            if is_valid_position(placements, None, x, y, w, h, map_w, map_h, blocked, chunk_size):
                placements.append(Placement(name, x, y, w, h))
                # Enforce reserved free space constraints
                if not free_capacity_ok(placements, map_w, map_h, blocked, reserve_empty_chunks, reserve_free_2x2, reserve_free_1x1, chunk_size):
                    placements.pop()
                    continue
                placed = True
                break
        if not placed:
            return None
    return placements

def optimize_layout(
    inventory_items: Dict[str, dict],
    map_w: int,
    map_h: int,
    blocked_tiles: Iterable[Tuple[int,int]],
    spots: List[dict],
    chunk_size: int = 4,
    restarts: int = 12,
    steps: int = 3500,
    seed: Optional[int] = None,
    reserve_empty_chunks: int = 0,
    reserve_free_2x2: int = 0,
    reserve_free_1x1: int = 0,
    on_progress: Optional[Callable[[int,int,float], None]] = None,
) -> Tuple[List[Placement], float, List[str]]:
    """
    Heuristic meta optimizer:
    - packs all instances from inventory (count>0 and w/h set)
    - simulated annealing moves (single-object relocation) using incremental scoring
    Returns: (best_placements, best_score, warnings)

    warnings: list of item names skipped because footprint missing.
    """
    rng = random.Random(seed)
    blocked = set((int(x),int(y)) for x,y in blocked_tiles)
    # Validate reserved space requirements against the empty map
    if reserve_empty_chunks < 0 or reserve_free_2x2 < 0 or reserve_free_1x1 < 0:
        raise ValueError("Valores de reserva não podem ser negativos.")
    max_chunks, max_2x2, max_1x1 = free_capacity_counts([], map_w, map_h, blocked, chunk_size)
    if reserve_empty_chunks > max_chunks:
        raise ValueError(f"Reserva de chunks 4x4 vazios ({reserve_empty_chunks}) excede o máximo possível ({max_chunks}) dado o mapa/bloqueios.")
    if reserve_free_2x2 > max_2x2:
        raise ValueError(f"Reserva de espaços 2x2 livres ({reserve_free_2x2}) excede o máximo possível ({max_2x2}) dado o mapa/bloqueios.")
    if reserve_free_1x1 > max_1x1:
        raise ValueError(f"Reserva de espaços 1x1 livres ({reserve_free_1x1}) excede o máximo possível ({max_1x1}) dado o mapa/bloqueios.")

    pair_bonus = build_pair_bonus(spots)

    # Build instances list
    instances: List[Tuple[str,int,int]] = []
    warnings: List[str] = []
    for name, info in inventory_items.items():
        cnt = int(info.get("count", 0) or 0)
        w = info.get("w", None)
        h = info.get("h", None)
        if cnt <= 0:
            continue
        if w is None or h is None:
            warnings.append(name)
            continue
        w = int(w); h = int(h)
        if w <= 0 or h <= 0:
            warnings.append(name)
            continue
        for _ in range(cnt):
            instances.append((name,w,h))

    if not instances:
        return [], 0.0, warnings

    def compute_total(placements: List[Placement]) -> float:
        return score_layout(placements, pair_bonus)

    def delta_move(placements: List[Placement], idx: int, new_x: int, new_y: int) -> float:
        """Incremental score delta if placements[idx] moved to (new_x,new_y)."""
        p_old = placements[idx]
        p_new = Placement(p_old.name, new_x, new_y, p_old.w, p_old.h)
        d = 0.0
        for j, other in enumerate(placements):
            if j == idx:
                continue
            key = (p_old.name, p_old.name) if p_old.name == other.name else tuple(sorted((p_old.name, other.name)))
            bonus = pair_bonus.get(key, 0.0)
            if bonus == 0.0:
                continue
            old_touch = rects_touch_8(p_old, other)
            new_touch = rects_touch_8(p_new, other)
            if old_touch == new_touch:
                continue
            d += bonus if new_touch else -bonus
        return d

    best_global: Optional[List[Placement]] = None
    best_global_score: float = -1e18

    # Precompute candidate positions by footprint to speed proposals
    cand_cache: Dict[Tuple[int,int], List[Tuple[int,int]]] = {}
    def get_cands(w:int,h:int):
        key=(w,h)
        if key not in cand_cache:
            cand_cache[key]=_candidate_positions(map_w,map_h,w,h,blocked,chunk_size)
        return cand_cache[key]

    for r in range(restarts):
        placements = _initial_pack(instances, map_w, map_h, blocked, chunk_size, rng, reserve_empty_chunks, reserve_free_2x2, reserve_free_1x1)
        if placements is None:
            continue

        current_score = compute_total(placements)
        best_local = list(placements)
        best_local_score = current_score

        # simulated annealing: move random item to random candidate pos
        # temperature scaled so that small negative moves sometimes accepted early
        temp0 = 5.0
        temp_end = 0.15
        for t in range(steps):
            frac = t / max(1, steps-1)
            temp = temp0 * (1 - frac) + temp_end * frac

            idx = rng.randrange(len(placements))
            p = placements[idx]
            cands = get_cands(p.w, p.h)
            if not cands:
                continue
            new_x, new_y = cands[rng.randrange(len(cands))]
            if new_x == p.x and new_y == p.y:
                continue
            if not is_valid_position(placements, idx, new_x, new_y, p.w, p.h, map_w, map_h, blocked, chunk_size):
                continue

            d = delta_move(placements, idx, new_x, new_y)
            accept = False
            if d >= 0:
                accept = True
            else:
                # probabilistic acceptance
                prob = math.exp(d / max(1e-9, temp))
                if rng.random() < prob:
                    accept = True

            if accept:
                placements[idx] = Placement(p.name, new_x, new_y, p.w, p.h)
                current_score += d
                if current_score > best_local_score:
                    best_local_score = current_score
                    best_local = list(placements)

            if on_progress and (t % 200 == 0 or t == steps-1):
                on_progress(r, t, best_local_score)

        if best_local_score > best_global_score:
            best_global_score = best_local_score
            best_global = best_local

        if on_progress:
            on_progress(r, steps, best_global_score)

    if best_global is None:
        return [], 0.0, warnings

    return best_global, best_global_score, warnings
# =========================
# v6 "META" optimizer (stronger + consistent)
# =========================

def _compactness_area(placements: list[Placement]) -> int:
    """Secondary criterion: smaller bounding box (more agrupado)."""
    if not placements:
        return 0
    minx = min(p.x for p in placements)
    miny = min(p.y for p in placements)
    maxx = max(p.x + p.w for p in placements)
    maxy = max(p.y + p.h for p in placements)
    return (maxx - minx) * (maxy - miny)


def _better_score(a_score: float, a_comp: int, b_score: float, b_comp: int) -> bool:
    """Return True if (a_score,a_comp) is better than (b_score,b_comp)."""
    if a_score > b_score + 1e-9:
        return True
    if abs(a_score - b_score) <= 1e-9 and a_comp < b_comp:
        return True
    return False


def _pair_key(name1: str, name2: str) -> tuple[str, str]:
    return (name1, name2) if name1 <= name2 else (name2, name1)


def _pair_score(p: Placement, q: Placement, pair_bonus: dict[tuple[str, str], float]) -> float:
    if not rects_touch(p, q):
        return 0.0
    return float(pair_bonus.get(_pair_key(p.name, q.name), 0.0))


def _contrib_for(idx: int, placements: list[Placement], pair_bonus: dict[tuple[str, str], float]) -> float:
    p = placements[idx]
    s = 0.0
    for j, q in enumerate(placements):
        if j == idx:
            continue
        s += _pair_score(p, q, pair_bonus)
    return s


def _tiles_of(p: Placement) -> list[tuple[int, int]]:
    return [(x, y) for y in range(p.y, p.y + p.h) for x in range(p.x, p.x + p.w)]


def _in_single_chunk(x: int, y: int, w: int, h: int, chunk_size: int) -> bool:
    cx0 = x // chunk_size
    cy0 = y // chunk_size
    cx1 = (x + w - 1) // chunk_size
    cy1 = (y + h - 1) // chunk_size
    return (cx0 == cx1) and (cy0 == cy1)


def _snap_to_chunk(x: int, y: int, w: int, h: int, map_w: int, map_h: int, chunk_size: int) -> tuple[int, int]:
    # clamp
    x = max(0, min(int(x), map_w - w))
    y = max(0, min(int(y), map_h - h))
    ox = (x % chunk_size) + w - chunk_size
    if ox > 0:
        x -= ox
    oy = (y % chunk_size) + h - chunk_size
    if oy > 0:
        y -= oy
    x = max(0, min(int(x), map_w - w))
    y = max(0, min(int(y), map_h - h))
    return x, y


def _build_occ(placements: list[Placement], map_w: int, map_h: int, blocked: set[tuple[int, int]]):
    occ = [[False for _ in range(map_w)] for _ in range(map_h)]
    for (bx, by) in blocked:
        if 0 <= bx < map_w and 0 <= by < map_h:
            occ[by][bx] = True
    for p in placements:
        for (x, y) in _tiles_of(p):
            if 0 <= x < map_w and 0 <= y < map_h:
                occ[y][x] = True
    return occ


def _rect_free(occ, x: int, y: int, w: int, h: int) -> bool:
    for yy in range(y, y + h):
        row = occ[yy]
        for xx in range(x, x + w):
            if row[xx]:
                return False
    return True


def _set_rect(occ, x: int, y: int, w: int, h: int, val: bool):
    for yy in range(y, y + h):
        row = occ[yy]
        for xx in range(x, x + w):
            row[xx] = val


def _initial_pack_biased(objects, map_w, map_h, blocked, pair_bonus, reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size, rng):
    """Greedy-ish random pack: place larger things first, bias towards center and existing clusters."""
    # order by area desc
    objs = list(objects)
    objs.sort(key=lambda o: (-(o['w'] * o['h']), o['name']))

    placements: list[Placement] = []
    occupied_tiles: set[tuple[int, int]] = set()

    # precompute candidate positions by size
    all_pos_4x4 = [(x, y) for x in range(0, map_w - 3, chunk_size) for y in range(0, map_h - 3, chunk_size)]
    all_pos_2x2 = [(x, y) for x in range(0, map_w - 1) for y in range(0, map_h - 1) if _in_single_chunk(x, y, 2, 2, chunk_size)]
    all_pos_1x1 = [(x, y) for x in range(0, map_w) for y in range(0, map_h)]

    def pos_list(w, h):
        if w == 4 and h == 4:
            return all_pos_4x4
        if w == 2 and h == 2:
            return all_pos_2x2
        return all_pos_1x1

    # helper to check local validity (fast) using occupied_tiles
    def can_place(x, y, w, h):
        if not _in_single_chunk(x, y, w, h, chunk_size):
            return False
        # bounds assumed by callers
        for yy in range(y, y + h):
            for xx in range(x, x + w):
                if (xx, yy) in blocked or (xx, yy) in occupied_tiles:
                    return False
        return True

    for obj in objs:
        w = int(obj['w']); h = int(obj['h'])
        # sample candidates
        candidates = pos_list(w, h)
        if not candidates:
            continue

        # bias: positions nearer center get more chances
        cx = (map_w - w) / 2
        cy = (map_h - h) / 2

        best_xy = None
        best_gain = -1e18

        # try more candidates for small items
        tries = 250 if w == 1 else 180 if w == 2 else 120

        for _ in range(tries):
            x, y = candidates[rng.randrange(len(candidates))]
            # small random jitter (only for 1x1/2x2), then snap
            if w != 4:
                x = max(0, min(map_w - w, x + rng.randint(-2, 2)))
                y = max(0, min(map_h - h, y + rng.randint(-2, 2)))
            x, y = _snap_to_chunk(x, y, w, h, map_w, map_h, chunk_size)
            if not (0 <= x <= map_w - w and 0 <= y <= map_h - h):
                continue
            if not can_place(x, y, w, h):
                continue

            # temporary placement
            p_new = Placement(obj['name'], x, y, w, h)
            # compute marginal gain against existing placements
            gain = 0.0
            for q in placements:
                gain += _pair_score(p_new, q, pair_bonus)

            # mild bias towards compactness (doesn't override score)
            dist = abs(x - cx) + abs(y - cy)
            gain -= 0.001 * dist

            # check reserve constraint if we accept this placement
            # (cheap because map <= few hundred tiles)
            new_tiles = [(xx, yy) for yy in range(y, y + h) for xx in range(x, x + w)]
            for t in new_tiles:
                occupied_tiles.add(t)
            ok = free_capacity_ok(map_w, map_h, occupied_tiles, blocked, reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size)
            for t in new_tiles:
                occupied_tiles.remove(t)

            if not ok:
                continue

            if gain > best_gain:
                best_gain = gain
                best_xy = (x, y)

        if best_xy is None:
            # couldn't place this object now
            continue

        x, y = best_xy
        p = Placement(obj['name'], x, y, w, h)
        placements.append(p)
        for t in _tiles_of(p):
            occupied_tiles.add(t)

    return placements


def _anneal_meta(
    placements: list[Placement],
    map_w: int,
    map_h: int,
    blocked: set[tuple[int, int]],
    pair_bonus: dict[tuple[str, str], float],
    reserve_clean_chunks: int,
    reserve_2x2: int,
    reserve_1x1: int,
    chunk_size: int,
    steps: int,
    start_temp: float,
    end_temp: float,
    rng: random.Random,
    on_progress=None,
    phase_name: str = ""
):
    # local mutable
    placements = [Placement(p.name, p.x, p.y, p.w, p.h, p.category) for p in placements]
    occ = _build_occ(placements, map_w, map_h, blocked)
    occupied_tiles = set()
    for p in placements:
        for t in _tiles_of(p):
            occupied_tiles.add(t)

    cur_score = score_layout(placements, pair_bonus)
    cur_comp = _compactness_area(placements)
    best_score = cur_score
    best_comp = cur_comp
    best = [Placement(p.name, p.x, p.y, p.w, p.h, p.category) for p in placements]

    n = len(placements)
    if n == 0 or steps <= 0:
        return best, best_score

    def temp_at(t):
        # exponential schedule
        if steps <= 1:
            return end_temp
        alpha = t / (steps - 1)
        return start_temp * ((end_temp / start_temp) ** alpha)

    # precompute all valid candidate positions by size
    pos_cache = {}
    def candidates_for(w, h):
        key = (w, h)
        if key in pos_cache:
            return pos_cache[key]
        if w == 4 and h == 4:
            lst = [(x, y) for x in range(0, map_w - 3, chunk_size) for y in range(0, map_h - 3, chunk_size)]
        elif w == 2 and h == 2:
            lst = [(x, y) for x in range(0, map_w - 1) for y in range(0, map_h - 1) if _in_single_chunk(x, y, 2, 2, chunk_size)]
        else:
            lst = [(x, y) for x in range(0, map_w) for y in range(0, map_h)]
        pos_cache[key] = lst
        return lst

    def try_move(i: int, newx: int, newy: int):
        nonlocal cur_score, cur_comp, best_score, best_comp, best
        p = placements[i]
        newx, newy = _snap_to_chunk(newx, newy, p.w, p.h, map_w, map_h, chunk_size)
        if not (0 <= newx <= map_w - p.w and 0 <= newy <= map_h - p.h):
            return False
        if not _in_single_chunk(newx, newy, p.w, p.h, chunk_size):
            return False

        # temporarily clear old rect in occ
        _set_rect(occ, p.x, p.y, p.w, p.h, False)
        if not _rect_free(occ, newx, newy, p.w, p.h):
            _set_rect(occ, p.x, p.y, p.w, p.h, True)
            return False

        # reserve constraint check with temporary occupied_tiles update
        old_tiles = _tiles_of(p)
        new_p = Placement(p.name, newx, newy, p.w, p.h, p.category)
        new_tiles = _tiles_of(new_p)
        for t in old_tiles:
            occupied_tiles.discard(t)
        for t in new_tiles:
            occupied_tiles.add(t)
        ok = free_capacity_ok(map_w, map_h, occupied_tiles, blocked, reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size)

        # revert temporary occupied update (we will re-apply if accepted)
        for t in new_tiles:
            occupied_tiles.discard(t)
        for t in old_tiles:
            occupied_tiles.add(t)

        if not ok:
            _set_rect(occ, p.x, p.y, p.w, p.h, True)
            return False

        # compute delta score
        old_contrib = 0.0
        new_contrib = 0.0
        for j, q in enumerate(placements):
            if j == i:
                continue
            old_contrib += _pair_score(p, q, pair_bonus)
            new_contrib += _pair_score(new_p, q, pair_bonus)
        delta = new_contrib - old_contrib
        new_score = cur_score + delta

        # accept?
        tcur = _anneal_meta._current_temp
        if delta >= 0.0:
            accept = True
        else:
            accept = (rng.random() < math.exp(delta / max(1e-9, tcur)))
        if not accept:
            _set_rect(occ, p.x, p.y, p.w, p.h, True)
            return False

        # apply: update occ
        _set_rect(occ, newx, newy, p.w, p.h, True)

        # apply occupied_tiles
        for t in old_tiles:
            occupied_tiles.discard(t)
        for t in new_tiles:
            occupied_tiles.add(t)

        placements[i] = new_p
        cur_score = new_score
        cur_comp = _compactness_area(placements)
        if _better_score(cur_score, cur_comp, best_score, best_comp):
            best_score, best_comp = cur_score, cur_comp
            best = [Placement(pp.name, pp.x, pp.y, pp.w, pp.h, pp.category) for pp in placements]
        return True

    def try_swap(i: int, j: int):
        nonlocal cur_score, cur_comp, best_score, best_comp, best
        if i == j:
            return False
        p = placements[i]
        q = placements[j]
        # propose swapping top-left positions
        p2 = Placement(p.name, q.x, q.y, p.w, p.h, p.category)
        q2 = Placement(q.name, p.x, p.y, q.w, q.h, q.category)

        # snap both
        p2x, p2y = _snap_to_chunk(p2.x, p2.y, p2.w, p2.h, map_w, map_h, chunk_size)
        q2x, q2y = _snap_to_chunk(q2.x, q2.y, q2.w, q2.h, map_w, map_h, chunk_size)
        p2 = Placement(p2.name, p2x, p2y, p2.w, p2.h, p2.category)
        q2 = Placement(q2.name, q2x, q2y, q2.w, q2.h, q2.category)

        if not (0 <= p2.x <= map_w - p2.w and 0 <= p2.y <= map_h - p2.h):
            return False
        if not (0 <= q2.x <= map_w - q2.w and 0 <= q2.y <= map_h - q2.h):
            return False
        if not _in_single_chunk(p2.x, p2.y, p2.w, p2.h, chunk_size):
            return False
        if not _in_single_chunk(q2.x, q2.y, q2.w, q2.h, chunk_size):
            return False

        # clear both in occ
        _set_rect(occ, p.x, p.y, p.w, p.h, False)
        _set_rect(occ, q.x, q.y, q.w, q.h, False)

        if not _rect_free(occ, p2.x, p2.y, p2.w, p2.h) or not _rect_free(occ, q2.x, q2.y, q2.w, q2.h):
            _set_rect(occ, p.x, p.y, p.w, p.h, True)
            _set_rect(occ, q.x, q.y, q.w, q.h, True)
            return False

        # reserve constraint check
        old_tiles_p = _tiles_of(p)
        old_tiles_q = _tiles_of(q)
        new_tiles_p = _tiles_of(p2)
        new_tiles_q = _tiles_of(q2)

        for t in old_tiles_p:
            occupied_tiles.discard(t)
        for t in old_tiles_q:
            occupied_tiles.discard(t)
        for t in new_tiles_p:
            occupied_tiles.add(t)
        for t in new_tiles_q:
            occupied_tiles.add(t)

        ok = free_capacity_ok(map_w, map_h, occupied_tiles, blocked, reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size)

        # revert for now
        for t in new_tiles_p:
            occupied_tiles.discard(t)
        for t in new_tiles_q:
            occupied_tiles.discard(t)
        for t in old_tiles_p:
            occupied_tiles.add(t)
        for t in old_tiles_q:
            occupied_tiles.add(t)

        if not ok:
            _set_rect(occ, p.x, p.y, p.w, p.h, True)
            _set_rect(occ, q.x, q.y, q.w, q.h, True)
            return False

        # delta score: only pairs touching i/j
        old = 0.0
        new = 0.0
        for k, r in enumerate(placements):
            if k == i or k == j:
                continue
            old += _pair_score(p, r, pair_bonus)
            old += _pair_score(q, r, pair_bonus)
            new += _pair_score(p2, r, pair_bonus)
            new += _pair_score(q2, r, pair_bonus)
        old += _pair_score(p, q, pair_bonus)
        new += _pair_score(p2, q2, pair_bonus)
        delta = new - old
        new_score = cur_score + delta

        tcur = _anneal_meta._current_temp
        if delta >= 0.0:
            accept = True
        else:
            accept = (rng.random() < math.exp(delta / max(1e-9, tcur)))
        if not accept:
            _set_rect(occ, p.x, p.y, p.w, p.h, True)
            _set_rect(occ, q.x, q.y, q.w, q.h, True)
            return False

        # apply in occ
        _set_rect(occ, p2.x, p2.y, p2.w, p2.h, True)
        _set_rect(occ, q2.x, q2.y, q2.w, q2.h, True)

        # apply occupied_tiles
        for t in old_tiles_p:
            occupied_tiles.discard(t)
        for t in old_tiles_q:
            occupied_tiles.discard(t)
        for t in new_tiles_p:
            occupied_tiles.add(t)
        for t in new_tiles_q:
            occupied_tiles.add(t)

        placements[i] = p2
        placements[j] = q2
        cur_score = new_score
        cur_comp = _compactness_area(placements)
        if _better_score(cur_score, cur_comp, best_score, best_comp):
            best_score, best_comp = cur_score, cur_comp
            best = [Placement(pp.name, pp.x, pp.y, pp.w, pp.h, pp.category) for pp in placements]
        return True

    # store temp to make it accessible inside nested funcs without recomputing
    _anneal_meta._current_temp = start_temp

    for t in range(steps):
        _anneal_meta._current_temp = temp_at(t)

        # move selection
        r = rng.random()
        if r < 0.62:
            # random move
            i = rng.randrange(n)
            p = placements[i]
            lst = candidates_for(p.w, p.h)
            x, y = lst[rng.randrange(len(lst))]
            # slight jitter
            if p.w != 4:
                x += rng.randint(-2, 2)
                y += rng.randint(-2, 2)
            try_move(i, x, y)

        elif r < 0.87 and n >= 2:
            # swap
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i != j:
                try_swap(i, j)

        else:
            # targeted: pick low-contrib and try to stick near a good partner
            i = rng.randrange(n)
            # occasionally compute true low contrib (cost O(n))
            if rng.random() < 0.35:
                # pick among a few random items
                cand = [rng.randrange(n) for _ in range(min(6, n))]
                cand.sort(key=lambda idx: _contrib_for(idx, placements, pair_bonus))
                i = cand[0]
            pi = placements[i]

            # find partners with positive bonus
            partners = []
            for j, pj in enumerate(placements):
                if j == i:
                    continue
                if pair_bonus.get(_pair_key(pi.name, pj.name), 0.0) > 0.0:
                    partners.append(j)
            if not partners:
                # fallback random move
                lst = candidates_for(pi.w, pi.h)
                x, y = lst[rng.randrange(len(lst))]
                try_move(i, x, y)
            else:
                j = partners[rng.randrange(len(partners))]
                pj = placements[j]
                # propose a position near pj
                # sample a few around its expanded bounding box
                ok = False
                for _ in range(12):
                    x = rng.randint(pj.x - pi.w - 1, pj.x + pj.w + 1)
                    y = rng.randint(pj.y - pi.h - 1, pj.y + pj.h + 1)
                    if try_move(i, x, y):
                        ok = True
                        break
                if not ok:
                    lst = candidates_for(pi.w, pi.h)
                    x, y = lst[rng.randrange(len(lst))]
                    try_move(i, x, y)

        if on_progress and (t % max(1, steps // 200) == 0):
            on_progress(100.0 * (t + 1) / steps, best_score)

    return best, best_score


def _refine_greedy(
    placements: list[Placement],
    map_w: int,
    map_h: int,
    blocked: set[tuple[int, int]],
    pair_bonus: dict[tuple[str, str], float],
    reserve_clean_chunks: int,
    reserve_2x2: int,
    reserve_1x1: int,
    chunk_size: int,
    passes: int = 2,
    on_progress=None,
):
    """Pente-fino final: tenta reposicionar cada peça para melhorar score (sem piorar)."""
    placements = [Placement(p.name, p.x, p.y, p.w, p.h, p.category) for p in placements]

    occ = _build_occ(placements, map_w, map_h, blocked)
    occupied_tiles = set()
    for p in placements:
        for t in _tiles_of(p):
            occupied_tiles.add(t)

    cur_score = score_layout(placements, pair_bonus)
    cur_comp = _compactness_area(placements)

    def candidates_for(p: Placement):
        if p.w == 4 and p.h == 4:
            return [(x, y) for x in range(0, map_w - 3, chunk_size) for y in range(0, map_h - 3, chunk_size)]
        if p.w == 2 and p.h == 2:
            return [(x, y) for x in range(0, map_w - 1) for y in range(0, map_h - 1) if _in_single_chunk(x, y, 2, 2, chunk_size)]
        return [(x, y) for x in range(0, map_w) for y in range(0, map_h)]

    n = len(placements)
    if n == 0:
        return placements, cur_score

    for pas in range(passes):
        improved_any = False
        for i in range(n):
            p = placements[i]

            # clear old
            _set_rect(occ, p.x, p.y, p.w, p.h, False)
            old_tiles = _tiles_of(p)
            for t in old_tiles:
                occupied_tiles.discard(t)

            best_p = p
            best_score = cur_score
            best_comp = cur_comp

            for (x, y) in candidates_for(p):
                x, y = _snap_to_chunk(x, y, p.w, p.h, map_w, map_h, chunk_size)
                if not (0 <= x <= map_w - p.w and 0 <= y <= map_h - p.h):
                    continue
                if not _in_single_chunk(x, y, p.w, p.h, chunk_size):
                    continue
                if not _rect_free(occ, x, y, p.w, p.h):
                    continue

                cand = Placement(p.name, x, y, p.w, p.h, p.category)
                new_tiles = _tiles_of(cand)
                for t in new_tiles:
                    occupied_tiles.add(t)
                ok = free_capacity_ok(map_w, map_h, occupied_tiles, blocked, reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size)
                for t in new_tiles:
                    occupied_tiles.discard(t)
                if not ok:
                    continue

                # compute delta against others (O(n))
                old_contrib = 0.0
                new_contrib = 0.0
                for j, q in enumerate(placements):
                    if j == i:
                        continue
                    old_contrib += _pair_score(p, q, pair_bonus)
                    new_contrib += _pair_score(cand, q, pair_bonus)
                delta = new_contrib - old_contrib
                sc = cur_score + delta

                # compactness tie-breaker
                # cheap compute: temporarily set
                placements[i] = cand
                comp = _compactness_area(placements)
                placements[i] = p

                if _better_score(sc, comp, best_score, best_comp):
                    best_score, best_comp = sc, comp
                    best_p = cand

            # apply best
            placements[i] = best_p
            # set rect
            _set_rect(occ, best_p.x, best_p.y, best_p.w, best_p.h, True)
            for t in _tiles_of(best_p):
                occupied_tiles.add(t)

            if best_p.x != p.x or best_p.y != p.y:
                improved_any = True
                cur_score = best_score
                cur_comp = best_comp

            if on_progress and (i % max(1, n // 50) == 0):
                on_progress((100.0 * (pas + (i + 1) / n) / passes), cur_score)

        if not improved_any:
            break

    return placements, cur_score


def _objects_from_inventory(inventory_items: list[dict]) -> tuple[list[dict], list[str]]:
    """Expand counts into individual objects. Returns (objects, warnings)"""
    objects = []
    warnings = []
    for item in inventory_items:
        name = item.get("name")
        w = item.get("w")
        h = item.get("h")
        if name is None:
            continue
        if w is None or h is None:
            warnings.append(str(name))
            continue
        w = int(w); h = int(h)
        if (w, h) not in [(1, 1), (2, 2), (4, 4)]:
            # v6: ignore any other footprint
            warnings.append(str(name))
            continue
        count = int(item.get("count", 0))
        if count <= 0:
            continue
        for _ in range(count):
            objects.append({"name": name, "w": w, "h": h, "category": item.get("category")})
    return objects, warnings


def optimize_layout_meta(
    map_w: int,
    map_h: int,
    inventory_items: list[dict],
    blocked: set[tuple[int, int]] | None = None,
    reserve_clean_chunks: int = 0,
    reserve_2x2: int = 0,
    reserve_1x1: int = 0,
    chunk_size: int = 4,
    restarts: int = 14,
    steps: int | None = None,
    seed: int | None = None,
    on_progress=None,
):
    """Heavy optimizer (multi-start + annealing + refinement)."""
    blocked = blocked or set()
    pair_bonus = build_pair_bonus(SPOTS)
    objects, warnings = _objects_from_inventory(inventory_items)

    if not objects:
        return [], 0.0, warnings

    # scale steps with object count if not given
    if steps is None:
        n = len(objects)
        # heavier than v5
        steps = max(12000, min(120000, 2500 + 900 * n))

    rng = random.Random(seed if seed is not None else random.randrange(1_000_000_000))

    best_global = []
    best_score = -1e18
    best_comp = 10**18

    for r in range(restarts):
        # make a fresh RNG stream per restart
        rrng = random.Random(rng.randrange(1_000_000_000))
        base = _initial_pack_biased(objects, map_w, map_h, blocked, pair_bonus,
                                   reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size, rrng)

        # if we couldn't place everything, still try optimizing what we placed
        # (GUI will show that some items didn't fit)

        if on_progress:
            on_progress(0.0, max(0.0, best_score if best_score > -1e17 else 0.0))

        # anneal
        def prog(pct, sc):
            # map restart progress into 0..100
            if on_progress:
                # 80% budget for anneal, 20% for refine
                overall = (r / restarts) * 100.0 + (pct / restarts) * 0.8
                on_progress(min(99.0, overall), max(sc, best_score if best_score > -1e17 else 0.0))

        annealed, sc = _anneal_meta(
            base,
            map_w,
            map_h,
            blocked,
            pair_bonus,
            reserve_clean_chunks,
            reserve_2x2,
            reserve_1x1,
            chunk_size,
            steps=steps,
            start_temp=6.0,
            end_temp=0.12,
            rng=rrng,
            on_progress=prog,
        )

        # refine
        refined, sc2 = _refine_greedy(
            annealed,
            map_w,
            map_h,
            blocked,
            pair_bonus,
            reserve_clean_chunks,
            reserve_2x2,
            reserve_1x1,
            chunk_size,
            passes=2,
            on_progress=None,
        )

        comp = _compactness_area(refined)
        if _better_score(sc2, comp, best_score, best_comp):
            best_score, best_comp = sc2, comp
            best_global = refined

        if on_progress:
            on_progress(min(99.0, ((r + 1) / restarts) * 100.0), best_score)

    if on_progress:
        on_progress(100.0, best_score)

    # warn if items couldn't be placed
    placed_names = [p.name for p in best_global]
    expected = len(objects)
    if len(best_global) < expected:
        warnings.append(f"(Aviso) Não couberam {expected - len(best_global)} item(ns) no mapa com as restrições atuais.")

    return best_global, float(max(0.0, best_score)), warnings


def improve_layout(
    current_placements: list[Placement],
    map_w: int,
    map_h: int,
    inventory_items: list[dict],
    blocked: set[tuple[int, int]] | None = None,
    reserve_clean_chunks: int = 0,
    reserve_2x2: int = 0,
    reserve_1x1: int = 0,
    chunk_size: int = 4,
    attempts: int = 10,
    steps_per_attempt: int | None = None,
    target_improve_factor: float | None = None,
    seed: int | None = None,
    on_progress=None,
):
    """Continue optimizing from the current layout (10 tentativas)."""
    blocked = blocked or set()
    pair_bonus = build_pair_bonus(SPOTS)

    objects, warnings = _objects_from_inventory(inventory_items)
    if not objects:
        return list(current_placements), 0.0, warnings

    if steps_per_attempt is None:
        n = max(1, len(current_placements))
        steps_per_attempt = max(5000, min(40000, 2000 + 500 * n))

    rng = random.Random(seed if seed is not None else random.randrange(1_000_000_000))

    # try to place missing items (if any)
    want = {}
    for o in objects:
        want[o['name']] = want.get(o['name'], 0) + 1
    have = {}
    for p in current_placements:
        have[p.name] = have.get(p.name, 0) + 1

    missing = []
    for o in objects:
        if have.get(o['name'], 0) > 0:
            have[o['name']] -= 1
        else:
            missing.append(o)

    placements = [Placement(p.name, p.x, p.y, p.w, p.h, p.category) for p in current_placements]

    if missing:
        # add missing by greedy scan
        occupied_tiles = set()
        for p in placements:
            for t in _tiles_of(p):
                occupied_tiles.add(t)

        for o in missing:
            w = int(o['w']); h = int(o['h'])
            placed = False
            # candidate positions (biased)
            cand = []
            if w == 4:
                cand = [(x, y) for x in range(0, map_w - 3, chunk_size) for y in range(0, map_h - 3, chunk_size)]
            elif w == 2:
                cand = [(x, y) for x in range(0, map_w - 1) for y in range(0, map_h - 1) if _in_single_chunk(x, y, 2, 2, chunk_size)]
            else:
                cand = [(x, y) for x in range(0, map_w) for y in range(0, map_h)]
            rng.shuffle(cand)
            for (x, y) in cand[:800]:
                x, y = _snap_to_chunk(x, y, w, h, map_w, map_h, chunk_size)
                if not (0 <= x <= map_w - w and 0 <= y <= map_h - h):
                    continue
                if not _in_single_chunk(x, y, w, h, chunk_size):
                    continue
                ok = True
                for yy in range(y, y + h):
                    for xx in range(x, x + w):
                        if (xx, yy) in blocked or (xx, yy) in occupied_tiles:
                            ok = False
                            break
                    if not ok:
                        break
                if not ok:
                    continue
                pnew = Placement(o['name'], x, y, w, h)
                # reserve constraint
                new_tiles = _tiles_of(pnew)
                for t in new_tiles:
                    occupied_tiles.add(t)
                cap_ok = free_capacity_ok(map_w, map_h, occupied_tiles, blocked,
                                          reserve_clean_chunks, reserve_2x2, reserve_1x1, chunk_size)
                if not cap_ok:
                    for t in new_tiles:
                        occupied_tiles.discard(t)
                    continue
                placements.append(pnew)
                placed = True
                break
            if not placed:
                warnings.append(f"(Aviso) Não coube no layout atual: {o['name']}")

    # baseline
    best = [Placement(p.name, p.x, p.y, p.w, p.h, p.category) for p in placements]
    best_score = score_layout(best, pair_bonus)
    best_comp = _compactness_area(best)

    target_score: float | None = None
    if target_improve_factor is not None:
        target_score = best_score * float(target_improve_factor)

    for att in range(attempts):
        arng = random.Random(rng.randrange(1_000_000_000))

        def prog(pct, sc):
            if on_progress:
                overall = (att / attempts) * 100.0 + (pct / attempts)
                on_progress(min(99.0, overall), max(sc, best_score))

        annealed, sc = _anneal_meta(
            best,
            map_w,
            map_h,
            blocked,
            pair_bonus,
            reserve_clean_chunks,
            reserve_2x2,
            reserve_1x1,
            chunk_size,
            steps=steps_per_attempt,
            start_temp=2.8,
            end_temp=0.08,
            rng=arng,
            on_progress=prog,
        )

        refined, sc2 = _refine_greedy(
            annealed,
            map_w,
            map_h,
            blocked,
            pair_bonus,
            reserve_clean_chunks,
            reserve_2x2,
            reserve_1x1,
            chunk_size,
            passes=1,
            on_progress=None,
        )

        comp = _compactness_area(refined)
        if _better_score(sc2, comp, best_score, best_comp):
            best_score, best_comp = sc2, comp
            best = refined

            if target_score is not None and best_score >= target_score:
                break

        if on_progress:
            on_progress(min(99.0, ((att + 1) / attempts) * 100.0), best_score)

    if on_progress:
        on_progress(100.0, best_score)

    return best, float(best_score), warnings
