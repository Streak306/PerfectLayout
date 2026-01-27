"""Layout optimization heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List


@dataclass(frozen=True)
class ItemSpec:
    name: str
    width: float
    height: float
    color: str


@dataclass
class Placement:
    x: float
    y: float
    rotated: bool


@dataclass
class LayoutResult:
    placements: List[Placement]
    score: float
    iterations: int
    temperature: float


@dataclass
class LayoutInput:
    room_width: float
    room_height: float
    items: List[ItemSpec]


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(value, max_value))


def _rect(item: ItemSpec, placement: Placement) -> tuple[float, float, float, float]:
    width, height = (item.height, item.width) if placement.rotated else (item.width, item.height)
    return (
        placement.x,
        placement.y,
        placement.x + width,
        placement.y + height,
    )


def _overlap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    overlap_w = max(0.0, min(x2, x4) - max(x1, x3))
    overlap_h = max(0.0, min(y2, y4) - max(y1, y3))
    return overlap_w * overlap_h


def _score_layout(layout: LayoutInput, placements: Iterable[Placement]) -> float:
    placements = list(placements)
    rects = [_rect(item, placement) for item, placement in zip(layout.items, placements)]
    room_area = layout.room_width * layout.room_height
    item_area = sum(item.width * item.height for item in layout.items)
    total_overlap = 0.0
    total_outside = 0.0
    for index, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        outside_w = max(0.0, -x1) + max(0.0, x2 - layout.room_width)
        outside_h = max(0.0, -y1) + max(0.0, y2 - layout.room_height)
        total_outside += outside_w * (y2 - y1) + outside_h * (x2 - x1)
        for other in rects[index + 1 :]:
            total_overlap += _overlap(rect, other)
    density = item_area / room_area if room_area > 0 else 0.0
    compactness_bonus = 1.0 + density
    penalty = total_overlap * 4.0 + total_outside * 6.0
    return item_area * compactness_bonus - penalty


def _initial_placements(layout: LayoutInput, rng: random.Random) -> List[Placement]:
    placements: List[Placement] = []
    for item in layout.items:
        rotated = rng.random() < 0.25
        width = item.height if rotated else item.width
        height = item.width if rotated else item.height
        x = rng.uniform(0.0, max(0.0, layout.room_width - width))
        y = rng.uniform(0.0, max(0.0, layout.room_height - height))
        placements.append(Placement(x=x, y=y, rotated=rotated))
    return placements


def optimize_layout(
    layout: LayoutInput,
    iterations: int = 2400,
    start_temperature: float = 2.2,
    seed: int | None = None,
) -> LayoutResult:
    rng = random.Random(seed)
    placements = _initial_placements(layout, rng)
    best = list(placements)
    best_score = _score_layout(layout, placements)
    temperature = start_temperature

    for step in range(iterations):
        idx = rng.randrange(len(placements))
        current = placements[idx]
        item = layout.items[idx]
        step_scale = max(layout.room_width, layout.room_height) * 0.15
        dx = rng.uniform(-step_scale, step_scale)
        dy = rng.uniform(-step_scale, step_scale)
        rotated = current.rotated if rng.random() > 0.15 else not current.rotated

        width = item.height if rotated else item.width
        height = item.width if rotated else item.height
        new_x = _clamp(current.x + dx, -width * 0.2, layout.room_width - width * 0.8)
        new_y = _clamp(current.y + dy, -height * 0.2, layout.room_height - height * 0.8)
        placements[idx] = Placement(x=new_x, y=new_y, rotated=rotated)

        new_score = _score_layout(layout, placements)
        score_delta = new_score - best_score
        accept_prob = math.exp(score_delta / max(0.001, temperature)) if score_delta < 0 else 1.0
        if new_score >= best_score or rng.random() < accept_prob:
            if new_score > best_score:
                best_score = new_score
                best = list(placements)
        else:
            placements[idx] = current

        temperature = max(0.12, temperature * 0.995)

    return LayoutResult(
        placements=best,
        score=best_score,
        iterations=iterations,
        temperature=temperature,
    )
