"""Local statistics, connected components and the edge-preserving smoothing that does the work.

`bilateral` is the core denoiser. It is a *range-weighted* mean: every pixel is replaced by the
average of the pixels around it whose colour is within a small Lab distance. That gives three
properties this task needs:

* grain and the fine part of the AI "mottle" get averaged away (many samples -> noise / sqrt(n)),
* a real edge survives, because pixels on the other side of it are down-weighted to zero,
* a real gradient survives unchanged, because a linear ramp's local mean *is* the ramp,
* and the estimate is unbiased, unlike a linear-model (guided) filter, which leaks neighbouring
  colours into small flat structures.

`label_components` then finds each connected flat block so it can be repainted with exactly one
colour - a local average cannot remove variation that is *larger* than its window, a block mean can.
"""

from __future__ import annotations

import numpy as np


def box_mean(a: np.ndarray, r: int) -> np.ndarray:
    """Mean over a (2r+1)^2 window, reflect edges, O(N) via an integral image."""
    a = np.asarray(a, dtype=np.float32)
    squeeze = a.ndim == 2
    if squeeze:
        a = a[..., None]
    h, w, _ = a.shape
    pad = np.pad(a, ((r, r), (r, r), (0, 0)), mode="reflect")
    # integral image with a leading zero row/column: cum[i, j] == sum of pad[:i, :j]
    cum = np.pad(np.cumsum(np.cumsum(pad, axis=0), axis=1), ((1, 0), (1, 0), (0, 0)))
    n = 2 * r + 1
    s = cum[n : n + h, n : n + w] - cum[0:h, n : n + w] - cum[n : n + h, 0:w] + cum[0:h, 0:w]
    out = s / float(n * n)
    return out[..., 0] if squeeze else out


def box_count(mask: np.ndarray, r: int) -> np.ndarray:
    """Number of True entries in a (2r+1)^2 window."""
    return box_mean(np.asarray(mask, dtype=np.float32), r) * float((2 * r + 1) ** 2)


def dilate(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary dilation with a square structuring element."""
    m = np.asarray(mask, dtype=bool)
    for _ in range(max(0, radius)):
        out = m.copy()
        out[1:, :] |= m[:-1, :]
        out[:-1, :] |= m[1:, :]
        out[:, 1:] |= m[:, :-1]
        out[:, :-1] |= m[:, 1:]
        m = out
    return m


def erode(mask: np.ndarray, radius: int = 1) -> np.ndarray:
    """Binary erosion with a square structuring element."""
    return ~dilate(~np.asarray(mask, dtype=bool), radius)


def label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labelling of a boolean mask.

    Run-length encoding + union-find: the per-pixel work is vectorised, only the run merge loop
    is python (a few tens of thousands of iterations on a megapixel mask).
    Returns (labels int32, component count); background is labelled -1.
    """
    m = np.asarray(mask, dtype=bool)
    h, w = m.shape
    labels = np.full((h, w), -1, dtype=np.int32)
    if not m.any():
        return labels, 0

    padded = np.zeros((h, w + 2), dtype=np.int8)
    padded[:, 1:-1] = m
    d = np.diff(padded, axis=1)  # +1 at a run start (x), -1 at a run end (x, exclusive)

    row_runs: dict[int, list[tuple[int, int]]] = {}
    bounds: list[tuple[int, int]] = []
    offsets: dict[int, int] = {}
    for r in np.nonzero(d.any(axis=1))[0].tolist():
        starts = np.nonzero(d[r] == 1)[0].tolist()
        ends = np.nonzero(d[r] == -1)[0].tolist()
        offsets[r] = len(bounds)
        row_runs[r] = list(zip(starts, ends))
        bounds.extend(row_runs[r])

    parent = list(range(len(bounds)))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:  # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in row_runs:
        prev = row_runs.get(r - 1)
        if not prev:
            continue
        cur = row_runs[r]
        o_cur, o_prev = offsets[r], offsets[r - 1]
        i = j = 0
        while i < len(cur) and j < len(prev):
            a, b = cur[i]
            a2, b2 = prev[j]
            if a2 >= b:
                j += 1
            elif a >= b2:
                i += 1
            else:  # the two runs overlap horizontally -> same component
                union(o_cur + i, o_prev + j)
                if b2 <= b:
                    j += 1
                else:
                    i += 1

    roots: dict[int, int] = {}
    for i in range(len(bounds)):
        root = find(i)
        if root not in roots:
            roots[root] = len(roots)
    for r, runs in row_runs.items():
        o = offsets[r]
        for k, (a, b) in enumerate(runs):
            labels[r, a:b] = roots[find(o + k)]
    return labels, len(roots)


def local_std(x: np.ndarray, r: int) -> np.ndarray:
    """Per-channel standard deviation over a (2r+1)^2 window (float32)."""
    x = np.asarray(x, dtype=np.float32)
    m1 = box_mean(x, r)
    m2 = box_mean(x * x, r)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def local_spread(lab: np.ndarray, r: int) -> np.ndarray:
    """Largest per-channel local std over the window, in Lab units."""
    return local_std(lab, r).max(-1)


def bilateral(
    lab: np.ndarray,
    valid: np.ndarray,
    radius: int,
    sigma_range: float,
    sigma_spatial: float | None = None,
    stride: int = 2,
) -> np.ndarray:
    """Range-weighted local mean of `lab` (H, W, 3), ignoring pixels with valid == False.

    radius        windows of (2*radius+1)^2 pixels
    sigma_range   colour sigma in Lab dE: bigger = flattens more, blurs soft shading
    sigma_spatial spatial sigma, default radius/2
    stride        sample every `stride`-th offset inside the window (an approximation that is
                  accurate while sigma_spatial >> stride)
    """
    lab = np.asarray(lab, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    h, w, _ = lab.shape
    if sigma_spatial is None:
        sigma_spatial = max(1.0, radius / 2.0)

    r = int(radius)
    step = max(1, stride)
    k = r // step  # symmetric grid, always containing the centre tap
    offsets = [
        (dy * step, dx * step)
        for dy in range(-k, k + 1)
        for dx in range(-k, k + 1)
        if (dy * step) ** 2 + (dx * step) ** 2 <= r * r
    ] or [(0, 0)]

    pad_lab = np.pad(lab, ((r, r), (r, r), (0, 0)), mode="edge")
    pad_ok = np.pad(valid, ((r, r), (r, r)), mode="constant")
    inv_range = np.float32(1.0 / (2.0 * sigma_range * sigma_range))
    inv_space = np.float32(1.0 / (2.0 * sigma_spatial * sigma_spatial))

    num = np.zeros_like(lab)
    den = np.zeros((h, w), dtype=np.float32)
    for dy, dx in offsets:
        sl = pad_lab[r + dy : r + dy + h, r + dx : r + dx + w]
        ok = pad_ok[r + dy : r + dy + h, r + dx : r + dx + w]
        diff = lab - sl
        d2 = diff[..., 0] * diff[..., 0] + diff[..., 1] * diff[..., 1] + diff[..., 2] * diff[..., 2]
        weight = np.exp(-d2 * inv_range)
        weight *= np.float32(np.exp(-(dy * dy + dx * dx) * inv_space))
        weight *= ok
        num += weight[..., None] * sl
        den += weight
    # if nothing similar enough was found around a pixel (isolated speck / 1 px feature),
    # keep the pixel instead of amplifying numerical noise, and never let the weights vanish
    return np.where(den[..., None] < 0.5, lab, num / np.maximum(den, 1e-3)[..., None])


def component_means(lab: np.ndarray, labels: np.ndarray, count: int) -> np.ndarray:
    """Mean colour of every labelled component -> (count, 3) float32, NaN-free."""
    flat_labels = labels.ravel()
    sel = flat_labels >= 0
    sums = np.zeros((count, 3), np.float64)
    idx = flat_labels[sel]
    for c in range(3):
        sums[:, c] = np.bincount(idx, weights=lab[..., c].ravel()[sel], minlength=count)
    counts = np.bincount(idx, minlength=count).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    return (sums / counts[:, None]).astype(np.float32)


def gradient_magnitude(x: np.ndarray) -> np.ndarray:
    """|d/dx| + |d/dy| of a scalar field (H, W)."""
    x = np.asarray(x, dtype=np.float32)
    gx = np.zeros_like(x)
    gy = np.zeros_like(x)
    gx[:, 1:] = x[:, 1:] - x[:, :-1]
    gy[1:, :] = x[1:, :] - x[:-1, :]
    return np.abs(gx) + np.abs(gy)


def robust_sigma_luma(gray: np.ndarray, valid: np.ndarray, flat_range: float = 3.0) -> float:
    """Per-pixel noise sigma from the 3x3 Laplacian, restricted to locally flat pixels.

    The median (not the mean) over flat pixels keeps edges from inflating the estimate.
    """
    gray = np.asarray(gray, dtype=np.float32)
    h, w = gray.shape
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:] - 4.0 * gray[1:-1, 1:-1]
    )
    win = np.stack([gray[i : i + h - 2, j : j + w - 2] for i in range(3) for j in range(3)], axis=-1)
    rng = win.max(-1) - win.min(-1)
    sel = (rng < flat_range) & valid[1:-1, 1:-1]
    if sel.sum() < 64:
        return 0.0
    return float(np.sqrt(np.pi / 2.0) / 6.0 * np.median(np.abs(lap[1:-1, 1:-1][sel])))


def masked_box_mean(x: np.ndarray, valid: np.ndarray, r: int) -> np.ndarray:
    """Box mean that ignores pixels outside `valid` (so transparent/garbage pixels cannot leak)."""
    m = np.asarray(valid, dtype=np.float32)
    num = box_mean(np.asarray(x, dtype=np.float32) * m[..., None], r)
    den = np.maximum(box_mean(m, r), 1e-6)
    return num / den[..., None]


def grow_labels(
    lab: np.ndarray,
    labels: np.ndarray,
    means: np.ndarray,
    seeds: np.ndarray,
    tol: float,
    iterations: int | None = None,
) -> np.ndarray:
    """Grow the seed components over pixels that are still within `tol` of the component mean.

    This is what turns a fragmented flat mask into whole blocks: growth stops at any colour step
    bigger than `tol`, so two different colours never merge, while a block's own interior - including
    the low-frequency mottle that the denoiser only reduced - is absorbed.
    """
    lab = np.asarray(lab, dtype=np.float32)
    h, w, _ = lab.shape
    out = np.where(seeds, np.asarray(labels, np.int32), -1).astype(np.int32)
    n = int(means.shape[0])
    if n == 0:
        return out
    iterations = iterations or max(16, min(48, h // 32 + 8))
    pad = np.empty((h + 2, w + 2), np.int32)
    for _ in range(iterations):
        unknown = out < 0
        if not unknown.any():
            break
        pad.fill(-1)
        pad[1:-1, 1:-1] = out
        best_d = np.full((h, w), np.inf, np.float32)
        best_l = np.full((h, w), -1, np.int32)
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = pad[1 + dy : 1 + dy + h, 1 + dx : 1 + dx + w]
            ok = (nb >= 0) & unknown
            if not ok.any():
                continue
            d = np.abs(lab - means[np.clip(nb, 0, n - 1)]).max(-1)
            better = ok & (d < best_d)
            best_d = np.where(better, d, best_d)
            best_l = np.where(better, nb, best_l)
        grow = unknown & (best_d <= tol)
        n_grow = int(grow.sum())
        if n_grow == 0 or n_grow < 0.001 * grow.size:  # converged (or a few stragglers)
            break
        out = np.where(grow, best_l, out)
    return out


def merge_close_labels(
    labels: np.ndarray,
    means: np.ndarray,
    lab: np.ndarray,
    tol: float,
    passes: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge adjacent components whose mean colour differs by less than `tol` (Lab dE).

    Two pieces of the same visual block end up with slightly different means (each is the average
    of a different part of the mottle field); without this merge a repaint would leave 1-level
    plateaus. Colours separated by more than `tol` are never merged.
    """
    labels = np.asarray(labels, np.int32)
    means = np.asarray(means, np.float32)
    lab = np.asarray(lab, np.float32)
    n = int(means.shape[0])
    if n == 0:
        return labels, means
    h, w = labels.shape
    parent = list(range(n))

    def find(x: int) -> int:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for _ in range(passes):
        merged = False
        for dy, dx in ((0, 1), (1, 0)):
            a = labels[: h - dy or None, : w - dx or None]
            b = labels[dy:, dx:]
            m = (a >= 0) & (b >= 0) & (a != b)
            if not m.any():
                continue
            pairs = np.unique(np.stack([a[m].ravel(), b[m].ravel()], axis=1), axis=0)
            for u, v in pairs.tolist():
                ru, rv = find(u), find(v)
                if ru != rv and float(np.abs(means[ru] - means[rv]).max()) <= tol:
                    parent[rv] = ru
                    merged = True
        if not merged:
            break

    roots = np.array([find(i) for i in range(n)], np.int32)
    _, inv = np.unique(roots, return_inverse=True)
    new_labels = np.where(labels >= 0, inv[labels.clip(0)].astype(np.int32), np.int32(-1))
    new_n = int(inv.max()) + 1
    flat = new_labels.ravel()
    sel = flat >= 0
    idx = flat[sel]
    sums = np.zeros((new_n, 3), np.float64)
    for c in range(3):
        sums[:, c] = np.bincount(idx, weights=lab[..., c].ravel()[sel], minlength=new_n)
    counts = np.maximum(np.bincount(idx, minlength=new_n), 1).astype(np.float64)
    return new_labels, (sums / counts[:, None]).astype(np.float32)
