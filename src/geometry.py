from __future__ import annotations

from math import acos, degrees, hypot, sqrt


Point = tuple[float, float]
Point3 = tuple[float, float, float]


def clamp(value: float, low: float = -1.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def angle_degrees(a: Point | Point3, b: Point | Point3, c: Point | Point3) -> float:
    """Return angle ABC in degrees, robust to repeated landmarks."""
    ba = [a[i] - b[i] for i in range(min(len(a), len(b), len(c)))]
    bc = [c[i] - b[i] for i in range(min(len(a), len(b), len(c)))]
    norm_ba = sqrt(sum(v * v for v in ba))
    norm_bc = sqrt(sum(v * v for v in bc))
    if norm_ba == 0 or norm_bc == 0:
        return 0.0
    cosine = clamp(sum(x * y for x, y in zip(ba, bc, strict=False)) / (norm_ba * norm_bc))
    return degrees(acos(cosine))


def distance(a: Point | Point3, b: Point | Point3) -> float:
    if len(a) >= 3 and len(b) >= 3:
        return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
    return hypot(a[0] - b[0], a[1] - b[1])


def torso_lean_degrees(hip: Point, shoulder: Point) -> float:
    vertical = (hip[0], hip[1] - 1.0)
    return angle_degrees(vertical, hip, shoulder)


def sequence_smoothness(values: list[float]) -> float:
    if len(values) < 3:
        return 0.0
    second_diffs = [
        abs(values[index + 1] - 2 * values[index] + values[index - 1])
        for index in range(1, len(values) - 1)
    ]
    return sum(second_diffs) / len(second_diffs)


def normalized_delta(current: float, baseline: float, scale: float) -> float:
    if scale <= 0:
        return 0.0
    return clamp((current - baseline) / scale, -1.0, 1.0)
