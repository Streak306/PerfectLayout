"""Layout optimization heuristics."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Sequence


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


def _bounding_box(rects: Sequence[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    min_x = min(rect[0] for rect in rects)
    min_y = min(rect[1] for rect in rects)
    max_x = max(rect[2] for rect in rects)
    max_y = max(rect[3] for rect in rects)
    return min_x, min_y, max_x, max_y


def _rect_distance(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    dx = max(0.0, max(x1, x3) - min(x2, x4))
    dy = max(0.0, max(y1, y3) - min(y2, y4))
    return math.hypot(dx, dy)


def _score_layout(layout: LayoutInput, placements: Iterable[Placement]) -> float:
    placements = list(placements)
    rects = [_rect(item, placement) for item, placement in zip(layout.items, placements)]
    room_area = layout.room_width * layout.room_height
    item_area = sum(item.width * item.height for item in layout.items)
    total_overlap = 0.0
    total_outside = 0.0
    total_spacing_penalty = 0.0
    for index, rect in enumerate(rects):
        x1, y1, x2, y2 = rect
        outside_w = max(0.0, -x1) + max(0.0, x2 - layout.room_width)
        outside_h = max(0.0, -y1) + max(0.0, y2 - layout.room_height)
        total_outside += outside_w * (y2 - y1) + outside_h * (x2 - x1)
        for other in rects[index + 1 :]:
            total_overlap += _overlap(rect, other)
            distance = _rect_distance(rect, other)
            preferred_gap = min(layout.room_width, layout.room_height) * 0.03
            if distance < preferred_gap:
                total_spacing_penalty += (preferred_gap - distance) ** 2
    if rects:
        box = _bounding_box(rects)
        box_area = max(0.01, (box[2] - box[0]) * (box[3] - box[1]))
        compactness = item_area / box_area
    else:
        compactness = 0.0
    density = item_area / room_area if room_area > 0 else 0.0
    compactness_bonus = 1.0 + density + compactness
    spread_penalty = max(0.0, (1.0 - compactness)) * room_area * 0.35
    penalty = total_overlap * 5.0 + total_outside * 7.0 + total_spacing_penalty * 0.6 + spread_penalty
    return item_area * compactness_bonus - penalty


def _initial_placements(layout: LayoutInput, rng: random.Random) -> List[Placement]:
    placements: list[Placement | None] = [None] * len(layout.items)
    indices = sorted(
        range(len(layout.items)),
        key=lambda idx: layout.items[idx].width * layout.items[idx].height,
        reverse=True,
    )
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    for idx in indices:
        item = layout.items[idx]
        rotation_options = [
            (False, item.width, item.height),
            (True, item.height, item.width),
        ]
        rng.shuffle(rotation_options)
        placed = False
        for rotated, width, height in rotation_options:
            if cursor_x + width <= layout.room_width and cursor_y + height <= layout.room_height:
                x = cursor_x
                y = cursor_y
                placements[idx] = Placement(x=x, y=y, rotated=rotated)
                cursor_x += width
                row_height = max(row_height, height)
                placed = True
                break
        if not placed:
            cursor_x = 0.0
            cursor_y += row_height
            row_height = 0.0
            for rotated, width, height in rotation_options:
                if cursor_y + height <= layout.room_height and width <= layout.room_width:
                    x = cursor_x
                    y = cursor_y
                    placements[idx] = Placement(x=x, y=y, rotated=rotated)
                    cursor_x += width
                    row_height = max(row_height, height)
                    placed = True
                    break
        if not placed:
            rotated = rng.random() < 0.5
            width = item.height if rotated else item.width
            height = item.width if rotated else item.height
            x = rng.uniform(0.0, max(0.0, layout.room_width - width))
            y = rng.uniform(0.0, max(0.0, layout.room_height - height))
            placements[idx] = Placement(x=x, y=y, rotated=rotated)

    jitter = min(layout.room_width, layout.room_height) * 0.02
    finalized: List[Placement] = []
    for placement, item in zip(placements, layout.items):
        if placement is None:
            rotated = rng.random() < 0.25
            width = item.height if rotated else item.width
            height = item.width if rotated else item.height
            x = rng.uniform(0.0, max(0.0, layout.room_width - width))
            y = rng.uniform(0.0, max(0.0, layout.room_height - height))
            finalized.append(Placement(x=x, y=y, rotated=rotated))
        else:
            width = item.height if placement.rotated else item.width
            height = item.width if placement.rotated else item.height
            x = _clamp(placement.x + rng.uniform(-jitter, jitter), 0.0, layout.room_width - width)
            y = _clamp(placement.y + rng.uniform(-jitter, jitter), 0.0, layout.room_height - height)
            finalized.append(Placement(x=x, y=y, rotated=placement.rotated))
    return finalized


def _propose_move(
    layout: LayoutInput,
    placements: List[Placement],
    rng: random.Random,
    step_scale: float,
) -> tuple[tuple[int, int] | None, int | None, Placement | None]:
    if len(placements) > 1 and rng.random() < 0.18:
        first, second = rng.sample(range(len(placements)), 2)
        placements[first], placements[second] = placements[second], placements[first]
        return (first, second), None, None

    idx = rng.randrange(len(placements))
    current = placements[idx]
    item = layout.items[idx]
    dx = rng.uniform(-step_scale, step_scale)
    dy = rng.uniform(-step_scale, step_scale)
    rotated = current.rotated if rng.random() > 0.2 else not current.rotated
    width = item.height if rotated else item.width
    height = item.width if rotated else item.height
    new_x = _clamp(current.x + dx, -width * 0.15, layout.room_width - width * 0.85)
    new_y = _clamp(current.y + dy, -height * 0.15, layout.room_height - height * 0.85)
    placements[idx] = Placement(x=new_x, y=new_y, rotated=rotated)
    return None, idx, current


def optimize_layout(
    layout: LayoutInput,
    iterations: int = 2400,
    start_temperature: float = 2.2,
    seed: int | None = None,
) -> LayoutResult:
    rng = random.Random(seed)
    restart_count = max(2, min(6, len(layout.items) // 4 + 2))
    iterations_per_run = max(450, iterations // restart_count)
    best_overall: List[Placement] | None = None
    best_overall_score = float("-inf")
    temperature_floor = max(0.08, start_temperature * 0.05)

    for run in range(restart_count):
        placements = _initial_placements(layout, rng)
        current_score = _score_layout(layout, placements)
        best = list(placements)
        best_score = current_score
        temperature = start_temperature * (1.05 - 0.08 * run)
        stagnation = 0

        for step in range(iterations_per_run):
            step_scale = max(layout.room_width, layout.room_height) * (0.18 * (temperature / start_temperature))
            swap_indices, idx, previous = _propose_move(layout, placements, rng, step_scale)
            new_score = _score_layout(layout, placements)
            score_delta = new_score - current_score
            accept_prob = math.exp(score_delta / max(0.001, temperature)) if score_delta < 0 else 1.0
            if score_delta >= 0 or rng.random() < accept_prob:
                current_score = new_score
                if new_score > best_score:
                    best_score = new_score
                    best = list(placements)
                    stagnation = 0
                else:
                    stagnation += 1
            else:
                if swap_indices is not None:
                    first, second = swap_indices
                    placements[first], placements[second] = placements[second], placements[first]
                elif idx is not None:
                    placements[idx] = previous  # type: ignore[assignment]
                stagnation += 1

            if stagnation > 120:
                placements = _initial_placements(layout, rng)
                current_score = _score_layout(layout, placements)
                stagnation = 0

            temperature = max(temperature_floor, temperature * 0.994)

        if best_score > best_overall_score:
            best_overall_score = best_score
            best_overall = list(best)

    total_iterations = iterations_per_run * restart_count
    return LayoutResult(
        placements=best_overall or [],
        score=best_overall_score,
        iterations=total_iterations,
        temperature=temperature_floor,
    )
