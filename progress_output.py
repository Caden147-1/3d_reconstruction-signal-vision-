"""Bounded-work progress with honest estimates and readable redirected logs."""

from collections import deque
import math
import os
import shutil
import sys
import time

from tqdm import tqdm
from tqdm.utils import disp_len, disp_trim


def duration(seconds):
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class _ProgressStream:
    """Let a tee handle terminal redraws separately from its persistent log."""

    def __init__(self, stream, snapshot):
        self.stream = stream
        self.snapshot = snapshot
        self.source = object()

    def write(self, value):
        snapshot_writer = getattr(self.stream, 'write_progress_snapshot', None)
        if snapshot_writer is not None:
            return snapshot_writer(value, self.snapshot, source=self.source)
        writer = getattr(self.stream, 'write_progress', self.stream.write)
        return writer(value)

    def __getattr__(self, name):
        return getattr(self.stream, name)


class WorkProgress:
    """Report completed work, never make an interrupted operation reach 100%.

    ``update`` takes an absolute count. Estimates use measured throughput and
    remain unavailable until at least two advancing samples and one second
    of measured time have been observed.
    Repeated counts refresh elapsed time without claiming additional work.
    There is no background thread and only the caller renders output.
    """

    def __init__(self, label, total=None, unit='items', *, stream=None,
                 clock=time.monotonic, mininterval=1., log_interval=15., min_samples=2,
                 estimate=None):
        if total is not None and (not math.isfinite(total) or total < 0):
            raise ValueError('Progress total must be finite and nonnegative')
        self.label, self.total, self.unit = label, total, unit
        self.stream = sys.stderr if stream is None else stream
        self.clock = clock
        self.mininterval, self.log_interval = mininterval, log_interval
        self.min_samples = max(1, min_samples)
        self.estimate = (os.environ.get('FYP_PROGRESS_ESTIMATE', '1') != '0'
                         if estimate is None else bool(estimate))
        self.started = self.clock()
        self.completed = 0
        self.samples = deque([(self.started, 0)], maxlen=21)
        self.advances = 0
        self.last_render = -math.inf
        self.closed = False
        self.status = None
        self.interactive = bool(getattr(self.stream, 'isatty', lambda: False)())
        self.bar = None
        self.progress_stream = None
        self._render(force=True)

    @property
    def elapsed(self):
        return max(0., self.clock() - self.started)

    @property
    def eta(self):
        if not self.estimate:
            return None
        if self.total is None:
            return None
        if self.completed >= self.total:
            return 0.
        if self.advances < self.min_samples:
            return None
        started, count = self.samples[0]
        # Include time since the last completed item. A slow/currently blocked
        # item must make the estimate longer rather than count down to zero.
        elapsed = self.clock() - started
        advance = self.completed - count
        if elapsed < 1. or advance <= 0:
            return None
        return (self.total - self.completed) * elapsed / advance

    def update(self, completed):
        if self.closed:
            return
        if not math.isfinite(completed) or completed < self.completed:
            raise ValueError('Completed progress must be finite and monotonic')
        if self.total is not None and completed > self.total:
            # If metadata underestimated the work, the remaining bound is no
            # longer known. Do not display 100% repeatedly as more work arrives.
            self.total = None
        if completed > self.completed:
            self.advances += 1
            now = self.clock()
            # Parallel workers can complete a whole batch almost together.
            # Retain a time window rather than replacing it with one burst.
            if now - self.samples[-1][0] >= .25:
                self.samples.append((now, completed))
        self.completed = completed
        self._render()

    def refresh(self):
        self._render()

    def iterate(self, items):
        """Count an item after the consumer finishes it, including ``continue``."""
        for item in items:
            yield item
            self.update(self.completed + 1)

    def _terminal_layout(self, eta_text):
        """Reserve space for counts and ETA before the label or optional bar."""
        try:
            columns = os.get_terminal_size(self.stream.fileno()).columns
        except (AttributeError, OSError, ValueError):
            columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        columns = max(1, columns - 1)  # Avoid a terminal's last-column wrap.
        total = '?' if self.total is None else f'{self.total:g}'
        count = f'{self.completed:g}/{total}'
        elapsed = ('elapsed ' if columns >= 100 else 't ') + duration(self.elapsed)
        parts = [f'{count} {self.unit}', elapsed, eta_text]
        if self.status:
            parts.append(self.status)
        # At narrow widths, remove optional detail rather than cropping ETA.
        if len(' | '.join(parts)) + 8 > columns:
            parts[0] = count
        if len(' | '.join(parts)) + 8 > columns:
            parts.remove(elapsed)
        detail = ' | '.join(parts)
        bar_width = 10 if self.total is not None and columns - len(detail) >= 40 else 0
        reserved = bar_width + 3 if bar_width else 0
        label_width = max(0, columns - len(detail) - reserved - 3)
        label = self.label
        if disp_len(label) > label_width:
            label = disp_trim(label, max(0, label_width - 3)) + ('...' if label_width >= 3 else '')
        description = f'{label} | {detail}' if label else detail
        bar_format = '{desc} |{bar:10}|' if bar_width else '{desc}'
        return description, bar_format, columns

    def _render(self, force=False):
        now = self.clock()
        interval = self.mininterval if self.interactive else self.log_interval
        if not force and now - self.last_render < interval:
            return
        self.last_render = now
        estimate = self.eta
        if not self.estimate:
            eta_text = 'ETA disabled'
        elif self.closed and self.status != 'complete':
            eta_text = 'ETA unavailable'
        elif estimate is not None:
            eta_text = f'ETA ~{duration(estimate)}'
        elif self.total is None:
            eta_text = 'ETA unavailable'
        else:
            eta_text = 'ETA sampling'
        suffix = f'elapsed {duration(self.elapsed)} | {eta_text}'
        if self.status:
            suffix += f' | {self.status}'
        total = '?' if self.total is None else f'{self.total:g}'
        snapshot = (f'[progress] {self.label} | {self.completed:g}/{total} {self.unit}'
                    f' | {suffix}')
        if self.interactive:
            description, bar_format, columns = self._terminal_layout(eta_text)
            if self.bar is None:
                self.progress_stream = _ProgressStream(self.stream, snapshot)
                self.bar = tqdm(total=self.total, desc=description, unit=self.unit,
                                file=self.progress_stream, ncols=columns, dynamic_ncols=False,
                                mininterval=self.mininterval, ascii=True,
                                bar_format=bar_format)
            self.progress_stream.snapshot = snapshot
            self.bar.total = self.total
            self.bar.n = self.completed
            self.bar.ncols = columns
            self.bar.bar_format = bar_format
            self.bar.set_description_str(description, refresh=False)
            self.bar.refresh()
        else:
            text = snapshot + '\n'
            getattr(self.stream, 'write_progress', self.stream.write)(text)
            self.stream.flush()

    def close(self, status=None):
        if self.closed:
            return
        complete = self.total is not None and self.completed >= self.total
        self.status = status or ('complete' if complete else 'incomplete')
        if self.status == 'complete' and self.total is not None and not complete:
            self.status = 'incomplete'
        self.closed = True
        self._render(force=True)
        if self.bar is not None:
            self.bar.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close('failed' if exc_type else None)
