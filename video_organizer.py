"""
Video Organizer  (multiprocessing edition)
------------------------------------------
• Probes video resolutions in parallel using all available CPU cores
• Moves files into quality-bucketed subfolders (max 50 files each)
• Anything that can't be probed / moved → _UNPROCESSED folder with a CSV log

Resolution buckets : 144p · 240p · 360p · 480p · 720p · 1080p_FullHD
                     1440p_2K · 2160p_4K · 4K_and_above
Sub-folder naming  : 1080p_FullHD_part_01 / _part_02 / …

Requirements:
    pip install opencv-python tqdm

Usage:
    python video_organizer.py <input_folder> <output_folder>

    # Optionally override worker count (default = all logical cores):
    python video_organizer.py <input_folder> <output_folder> --workers 8
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import shutil
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

# ── Resolution probe ──────────────────────────────────────────────────────────
try:
    import cv2
except ImportError:
    print("ERROR: opencv-python is required.\n  pip install opencv-python")
    sys.exit(1)

# ── Constants ─────────────────────────────────────────────────────────────────
VIDEO_EXTENSIONS: set[str] = {
    ".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv",
    ".webm", ".m4v", ".mpeg", ".mpg", ".3gp", ".ts",
}

MAX_FILES_PER_FOLDER = 50

QUALITY_BUCKETS: list[tuple[int, str]] = [
    (144,  "144p"),
    (240,  "240p"),
    (360,  "360p"),
    (480,  "480p"),
    (720,  "720p"),
    (1080, "1080p_FullHD"),
    (1440, "1440p_2K"),
    (2160, "2160p_4K"),
]

UNPROCESSED_DIR = "_UNPROCESSED"


# ─────────────────────────────────────────────────────────────────────────────
# Worker function  (runs in a subprocess — no shared state allowed)
# ─────────────────────────────────────────────────────────────────────────────

def _probe_worker(path_str: str) -> tuple[str, int | None, int | None, str | None]:
    """Return (path_str, width, height, error_msg)."""
    try:
        cap = cv2.VideoCapture(path_str)
        if not cap.isOpened():
            return path_str, None, None, "cv2 could not open file"
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if w == 0 or h == 0:
            return path_str, None, None, "zero-dimension reported by cv2"
        return path_str, w, h, None
    except Exception as exc:
        return path_str, None, None, str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def classify(height: int) -> str:
    for threshold, label in QUALITY_BUCKETS:
        if height <= threshold:
            return label
    return "4K_and_above"


def collect_videos(folder: Path) -> list[Path]:
    results: list[Path] = []
    for root, _, files in os.walk(folder):
        for f in files:
            if Path(f).suffix.lower() in VIDEO_EXTENSIONS:
                results.append(Path(root) / f)
    return results


class FolderAllocator:
    """
    Thread-safe subfolder slot allocator.
    Hands out the next part-folder that still has room, creating it if needed.
    """

    def __init__(self, base: Path) -> None:
        self._base     = base
        self._counters: dict[str, int]            = {}
        self._counts:   dict[str, int]            = {}
        self._locks:    dict[str, threading.Lock] = {}
        self._global    = threading.Lock()

    def _lock_for(self, bucket: str) -> threading.Lock:
        with self._global:
            if bucket not in self._locks:
                self._locks[bucket] = threading.Lock()
            return self._locks[bucket]

    def next_slot(self, bucket: str) -> Path:
        lock = self._lock_for(bucket)
        with lock:
            part = self._counters.get(bucket, 1)
            while True:
                key    = f"{bucket}/{part}"
                folder = self._base / bucket / f"{bucket}_part_{part:02d}"
                folder.mkdir(parents=True, exist_ok=True)
                if key not in self._counts:
                    self._counts[key] = sum(1 for f in folder.iterdir() if f.is_file())
                if self._counts[key] < MAX_FILES_PER_FOLDER:
                    self._counts[key] += 1
                    self._counters[bucket] = part
                    return folder
                part += 1


def safe_dest(folder: Path, filename: str) -> Path:
    """Return a collision-free destination path inside folder."""
    dest = folder / filename
    if not dest.exists():
        return dest
    stem, suffix = Path(filename).stem, Path(filename).suffix
    i = 1
    while True:
        dest = folder / f"{stem}_{i}{suffix}"
        if not dest.exists():
            return dest
        i += 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def organise(input_folder: str, output_folder: str, workers: int) -> None:
    inp = Path(input_folder).resolve()
    out = Path(output_folder).resolve()

    if not inp.exists():
        print(f"[ERROR] Input folder not found: {inp}")
        sys.exit(1)

    out.mkdir(parents=True, exist_ok=True)

    # ── Logging ────────────────────────────────────────────────────────────
    log_path = out / f"organizer_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        filename=log_path, level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("organizer")

    # ── Collect ────────────────────────────────────────────────────────────
    print(f"\n  Scanning  {inp}")
    videos = collect_videos(inp)
    total  = len(videos)
    print(f"  Found {total} video file(s).\n")
    if not total:
        print("Nothing to do.")
        return

    cpu_count = mp.cpu_count()
    workers   = min(workers, cpu_count, total)
    print(f"  CPU cores available : {cpu_count}")
    print(f"  Worker processes    : {workers}\n")

    # ── Phase 1: parallel resolution probe ────────────────────────────────
    print("[ Phase 1 / 2 ]  Probing resolutions in parallel …")
    t0 = time.perf_counter()

    probe_results: list[tuple[str, int | None, int | None, str | None]] = []
    chunksize = max(1, total // (workers * 4))

    with mp.Pool(processes=workers) as pool:
        with tqdm(total=total, unit="file", dynamic_ncols=True) as bar:
            for result in pool.imap_unordered(
                _probe_worker, [str(p) for p in videos], chunksize=chunksize
            ):
                probe_results.append(result)
                bar.update()

    probe_time = time.perf_counter() - t0
    print(f"  Probed {total} files in {probe_time:.1f}s\n")

    # ── Phase 2: move files ────────────────────────────────────────────────
    print("[ Phase 2 / 2 ]  Moving files …")
    allocator  = FolderAllocator(out)
    unproc_dir = out / UNPROCESSED_DIR
    unproc_dir.mkdir(parents=True, exist_ok=True)

    stats:  dict[str, int]        = {}
    unproc: list[tuple[str, str]] = []   # (filename, reason)

    t1 = time.perf_counter()
    for path_str, w, h, err in tqdm(probe_results, unit="file", dynamic_ncols=True):
        src = Path(path_str)
        try:
            if err or h is None:
                reason = err or "unknown probe error"
                dest   = safe_dest(unproc_dir, src.name)
                shutil.move(str(src), str(dest))
                unproc.append((src.name, reason))
                log.warning("UNPROCESSED  %s  —  %s", src.name, reason)
                continue

            bucket      = classify(h)
            dest_folder = allocator.next_slot(bucket)
            dest        = safe_dest(dest_folder, src.name)
            shutil.move(str(src), str(dest))
            stats[bucket] = stats.get(bucket, 0) + 1
            log.info("MOVED  %s  →  %s  [%dx%d]", src.name, dest.parent.name, w, h)

        except Exception as exc:
            reason = str(exc)
            try:
                shutil.move(str(src), str(safe_dest(unproc_dir, src.name)))
            except Exception:
                pass
            unproc.append((src.name, reason))
            log.error("FAILED  %s  —  %s", src.name, reason)

    move_time = time.perf_counter() - t1

    # ── Unprocessed CSV report ─────────────────────────────────────────────
    if unproc:
        csv_path = unproc_dir / "unprocessed_report.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["filename", "reason"])
            writer.writerows(unproc)

    # ── Summary ────────────────────────────────────────────────────────────
    total_moved = sum(stats.values())
    divider = "─" * 44
    print(f"\n{divider}")
    print(f"  {'Quality':<24} {'Files':>8}")
    print(divider)
    for bucket, count in sorted(stats.items()):
        print(f"  {bucket:<24} {count:>8}")
    if unproc:
        print(f"  {'_UNPROCESSED':<24} {len(unproc):>8}")
    print(divider)
    print(f"  {'TOTAL':<24} {total_moved + len(unproc):>8}")
    print(f"\n  Probe time : {probe_time:>6.1f}s")
    print(f"  Move time  : {move_time:>6.1f}s")
    print(f"  Log        : {log_path}")
    if unproc:
        print(f"  Unprocessed: {unproc_dir / 'unprocessed_report.csv'}")
    print()


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mp.freeze_support()   # required on Windows when bundled as .exe

    parser = argparse.ArgumentParser(description="Organise video files by resolution.")
    parser.add_argument("input_folder",  help="Folder to scan (recursive)")
    parser.add_argument("output_folder", help="Destination root folder")
    parser.add_argument(
        "--workers", type=int, default=mp.cpu_count(),
        help=f"Parallel probe workers (default: {mp.cpu_count()} = all cores)",
    )
    args = parser.parse_args()

    organise(args.input_folder, args.output_folder, args.workers)
