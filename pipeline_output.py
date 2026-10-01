"""Compact console progress with a complete, buffered subprocess log on disk."""

from __future__ import annotations

from collections import deque
from datetime import datetime
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time

from tqdm import tqdm

from process_control import managed_process


VIEW_RE = re.compile(r"Processing view\s+(\d+)\s*/\s*(\d+)")
NATIVE_PREFIX_RE = re.compile(r"^([IWEF])\d{8}\s+\S+\s+\d+\s+[^\]]+\]\s*")
OUTPUT_RE = re.compile(r"Writing (photometric|geometric) output")
STAGE_RE = re.compile(r"^\[(?:\d+/\d+|preflight|finalize|cooldown)\]", re.I)
NUMBERED_STAGE_RE = re.compile(r"^\[\d+/\d+\]\s*(.+)")
ALERT_RE = re.compile(r"\b(?:error|warning|fatal|failed|failure|traceback)\b", re.I)


def duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class StageTimings:
    """Account for all runner time, including preflight and inter-stage waits."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = self.stage_started = clock()
        self.label = 'Preflight'
        self.stages = []

    def observe(self, line):
        match = NUMBERED_STAGE_RE.match(line.strip())
        if match:
            now = self.clock()
            self._append(now)
            self.label = match.group(1)
            self.stage_started = now

    def _append(self, now):
        self.stages.append({'stage': self.label,
                            'start_seconds': self.stage_started - self.started,
                            'elapsed_seconds': now - self.stage_started})

    def finish(self, status):
        now = self.clock()
        self._append(now)
        return {'schema_version': 1, 'status': status,
                'elapsed_seconds': now - self.started, 'stages': self.stages,
                'interpretation': 'Wall-clock time; waits belong to the preceding stage. '
                                  'The last stage also includes final reporting.'}


class CompactProgress:
    """Track one PatchMatch pass at a time; never present it as the whole-run ETA."""

    def __init__(self, *, stream=None, clock=time.monotonic, estimate=True, sample_views=2):
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.estimate = estimate
        self.sample_views = max(1, sample_views)
        self.interactive = self.stream.isatty()
        self.pass_number = 0
        self.phase = ""
        self.view = self.total = 0
        self.view_done = False
        self.view_started = self.pass_started = 0.0
        self.samples = deque(maxlen=max(20, self.sample_views))
        self.last_update = float("-inf")
        self.bar = None
        self.last_message = ""
        self.pending_severity = None
        self.last_console_output = float("-inf")
        self.repeated_warning = None
        self.warning_repeats = 0
        self.warning_reported_at = 0.0

    def clear(self):
        if self.bar is not None:
            self.bar.clear()

    def message(self, text):
        tqdm.write(text, file=self.stream)
        self.stream.flush()
        self.last_console_output = self.clock()

    def flush_warning_repeats(self):
        if self.warning_repeats:
            self.message(f"[warn] Previous warning repeated {self.warning_repeats} more times; see full log.")
            self.warning_repeats = 0
        self.warning_reported_at = self.clock()

    def progress(self, *, force=False):
        if not self.view:
            return
        now = self.clock()
        interval = 1.0 if self.interactive else 15.0
        if not force and now - self.last_update < interval:
            return
        completed = self.view if self.view_done else self.view - 1
        label = f"PatchMatch pass {self.pass_number}"
        if self.phase:
            label += f"/{self.phase}"
        eta = ""
        if self.estimate and len(self.samples) >= self.sample_views:
            average = sum(self.samples) / len(self.samples)
            remaining = average * (self.total - completed)
            if not self.view_done:
                remaining -= min(average, now - self.view_started)
            eta = f"ETA/pass ~{duration(remaining)}"
        elif self.estimate:
            eta = "ETA/pass: sampling"
        bar_format = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}{postfix}]"
        if self.interactive:
            if self.bar is None:
                self.bar = tqdm(total=self.total, desc=label, file=self.stream,
                                initial=completed, unit="view", ascii=True,
                                dynamic_ncols=True, mininterval=1., miniters=1,
                                bar_format=bar_format, postfix=eta, leave=True)
            else:
                self.bar.set_description_str(label, refresh=False)
                self.bar.set_postfix_str(eta, refresh=False)
                self.bar.update(completed - self.bar.n)
                self.bar.refresh()
        else:
            # Redirected output needs occasional complete lines, not terminal
            # cursor control. Use the same tqdm format without a live bar.
            text = tqdm.format_meter(completed, self.total, now - self.pass_started,
                                     prefix=label, ascii=True, ncols=120,
                                     bar_format=bar_format, postfix=eta)
            print(text, file=self.stream, flush=True)
        self.last_update = now
        self.last_console_output = now

    def finish_pass(self):
        self.progress(force=True)
        if self.bar is not None:
            self.bar.close()
            self.bar = None
        self.view = self.total = 0

    def close(self):
        self.flush_warning_repeats()
        self.finish_pass()

    def observe(self, line):
        text = line.strip()
        prefix = NATIVE_PREFIX_RE.match(text)
        severity = prefix.group(1) if prefix else None
        if prefix:
            text = text[prefix.end():]
            self.pending_severity = severity if severity in {"W", "E", "F"} else None
        if not text:
            return
        severity = self.pending_severity or severity
        self.pending_severity = None
        if (severity in {"W", "E", "F"} or ALERT_RE.search(text) or
                text.lower().startswith(("[warn]", "[warning]"))):
            is_warning = severity == "W" or text.lower().startswith(("[warn]", "[warning]"))
            if is_warning and text == self.repeated_warning:
                self.warning_repeats += 1
                if self.clock() - self.warning_reported_at >= 60:
                    self.flush_warning_repeats()
                return
            self.flush_warning_repeats()
            self.repeated_warning = text if is_warning else None
            self.message(f"[{severity}] {text}" if severity in {"W", "E", "F"} else text)
            return
        view = VIEW_RE.search(text)
        if view:
            index, total = map(int, view.groups())
            if not 1 <= index <= total:
                return
            now = self.clock()
            new_pass = (not self.view or index < self.view or total != self.total
                        or (index == 1 and self.view == 1 and self.view_done))
            if new_pass:
                if self.view:
                    self.finish_pass()
                self.pass_number += 1
                self.pass_started = now
                self.phase = ""
                self.samples.clear()
            elif index == self.view:
                return  # Duplicate native status line, not another completed view.
            self.view, self.total = index, total
            self.view_done = False
            self.view_started = now
            self.progress(force=new_pass)
            return
        output = OUTPUT_RE.search(text)
        if output and self.view:
            self.phase = output.group(1)
            if not self.view_done:
                self.samples.append(max(0.0, self.clock() - self.view_started))
                self.view_done = True
                self.progress(force=self.view == self.total)
            return
        if text.lower().startswith(("[coverage]", "[provenance]", "[load]",
                                    "[cooldown]", "[snapshot]", "[preflight]")):
            return  # Routine metadata remains available in the full log.
        if STAGE_RE.match(text):
            self.flush_warning_repeats()
            self.repeated_warning = None
            self.finish_pass()
            self.message(text)
        elif text.startswith("[") and text != self.last_message:
            self.message(text)
            self.last_message = text
        # Native option dumps, timestamps and separators remain in the log only.
        else:
            self.progress()


class ProcessMonitorError(RuntimeError):
    pass


def monitored_output(process, *, shell=False, output_timeout=0,
                     poll_seconds=0.25, shell_grace=30, exit_grace=5):
    """Read on a worker so silence/partial lines never block Ctrl-C or polling.

    None is a timer tick. A live computational child may legitimately be silent;
    only a stranded batch shell or an explicitly configured silence limit fails.
    """
    messages = queue.Queue(maxsize=1024)
    stopped = threading.Event()

    def publish(item):
        while not stopped.is_set():
            try:
                messages.put(item, timeout=0.1)
                return
            except queue.Full:
                pass

    def read():
        try:
            for line in process.stdout:
                publish(("line", line))
                if stopped.is_set():
                    return
        except Exception as exc:
            publish(("error", exc))
        finally:
            publish(("eof", None))

    reader = threading.Thread(target=read, name="subprocess-output", daemon=True)
    reader.start()
    last_output = time.monotonic()
    exited_at = alone_since = None
    eof = False
    try:
        while True:
            try:
                kind, value = messages.get(timeout=poll_seconds)
            except queue.Empty:
                kind, value = "tick", None
            now = time.monotonic()
            if kind == "line":
                last_output = now
                yield value
            elif kind == "error":
                raise ProcessMonitorError(f"Cannot read child output: {value}") from value
            elif kind == "eof":
                eof = True

            code = process.poll()
            if code is not None:
                if eof:
                    return
                exited_at = now if exited_at is None else exited_at
                if now - exited_at >= exit_grace and messages.empty():
                    raise ProcessMonitorError(
                        f"Process {process.pid} exited with code {code}, but its output "
                        "pipe is still open; terminating remaining descendants")
            else:
                if output_timeout and now - last_output >= output_timeout:
                    raise ProcessMonitorError(
                        f"No complete output line for {duration(now - last_output)}; "
                        "configured output timeout reached")
                count = getattr(process, "active_tree_processes", None)
                if shell and count is not None and count() == 1:
                    alone_since = now if alone_since is None else alone_since
                    if now - alone_since >= shell_grace:
                        raise ProcessMonitorError(
                            f"Batch shell {process.pid} has had no running child for "
                            f"{duration(now - alone_since)}; aborting abnormal wait")
                else:
                    alone_since = None
            if kind != "line":
                yield None
    finally:
        stopped.set()
        # managed_process kills the tree before closing stdout, releasing reads.


def stream_command(command, *, workspace, env, verbose=False, estimator=None,
                   estimate=True, sample_views=2, label="colmap",
                   heartbeat_seconds=60, output_timeout=None) -> int:
    """Drain stdout/stderr continuously; retain details even on failure or Ctrl-C."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    env = dict(env)
    env["FYP_COLMAP_RUN_ID"] = stamp
    if output_timeout is None:
        output_timeout = float(env.get("COLMAP_OUTPUT_TIMEOUT_SECONDS", "0"))
    if not math.isfinite(output_timeout) or output_timeout < 0:
        raise ValueError("COLMAP_OUTPUT_TIMEOUT_SECONDS must be finite and nonnegative")
    log_path = log_dir / f"{label}_{stamp}.log"
    display = CompactProgress(estimate=estimate, sample_views=sample_views)
    display.message(f"[log] Full output: {log_path}")
    tail = deque(maxlen=15)
    started = time.monotonic()
    timings = StageTimings()
    status = 'interrupted_or_failed'
    code = None
    failure = None
    try:
        # Buffered writes avoid a disk flush and terminal redraw for every line.
        with log_path.open("x", encoding="utf-8", buffering=64 * 1024) as log:
            with managed_process(command, cwd=str(workspace), env=env,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, errors="replace") as process:
                last_flush = last_output = last_heartbeat = time.monotonic()
                output = monitored_output(
                    process, shell=Path(command[0]).name.lower() in {"cmd", "cmd.exe"},
                    output_timeout=output_timeout)
                try:
                    for line in output:
                        now = time.monotonic()
                        if line is not None:
                            last_output = now
                            timings.observe(line)
                            log.write(line)
                            tail.append(line.rstrip())
                            if verbose:
                                print(line, end="", flush=True)
                                if estimator is not None:
                                    estimator.observe(line)
                            else:
                                display.observe(line)
                        if now - last_heartbeat >= heartbeat_seconds:
                            message = (f"[monitor] PID {process.pid} | {timings.label} | "
                                       f"elapsed {duration(now - started)} | "
                                       f"no output {duration(now - last_output)}")
                            # An active progress bar or recent message already
                            # shows liveness; only print a heartbeat when quiet.
                            if ((verbose and now - last_output >= heartbeat_seconds) or
                                    (not verbose and now - display.last_console_output >= heartbeat_seconds)):
                                display.flush_warning_repeats()
                                display.message(message)
                            log.write(message + "\n")
                            last_heartbeat = now
                        if now - last_flush >= 5:
                            log.flush()
                            last_flush = now
                    code = process.returncode
                    status = 'completed' if code == 0 else 'failed'
                except ProcessMonitorError as exc:
                    failure = str(exc)
                    code = process.poll() or 1
                    status = 'monitor_failed'
                    log.write(f"[error] {failure}\n")
                    display.message(f"[error] {failure}")
                finally:
                    output.close()
                    log.flush()
        display.close()
        if code:
            display.message(f"[error] {label} exited with code {code}. Last output:")
            for line in tail:
                display.message(line)
        else:
            display.message(f"[done] {label} | elapsed {duration(time.monotonic() - started)}")
        return code
    except KeyboardInterrupt:
        display.message(f"[interrupted] {label}; partial log: {log_path}")
        raise
    finally:
        display.close()
        report = timings.finish(status)
        report['log'] = str(log_path)
        report['exit_code'] = code
        if failure:
            report['failure_reason'] = failure
        try:
            timing_path = log_path.with_suffix('.timing.json')
            timing_path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
            if verbose:
                for row in report['stages']:
                    display.message(f"[timing] {row['stage']}: {duration(row['elapsed_seconds'])}")
        except OSError as exc:
            display.message(f'[warning] Could not save stage timings: {exc}')
        if estimator is not None:
            estimator.close()
