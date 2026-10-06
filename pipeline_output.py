"""Compact console progress with a complete, buffered subprocess log on disk."""

from __future__ import annotations

from collections import deque
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import queue
import re
import subprocess
import sys
import threading
import time
import traceback

from tqdm import tqdm
from progress_output import WorkProgress

from process_control import managed_process
from colmap_workspace_check import write_provenance
from workspace_lock import workspace_activity


CONSOLE = "main_console.log"
STATE = "main_run_state.json"


class Tee:
    def __init__(self, terminal, log, mutex, progress_state=None, clock=time.monotonic):
        self.terminal, self.log, self.mutex = terminal, log, mutex
        self.clock = clock
        self.progress_state = progress_state if progress_state is not None else {}

    def _clear_progress(self):
        state = self.progress_state
        terminal = state.pop('terminal', None)
        if terminal is not None:
            terminal.write('\r' + ' ' * state.get('width', 0) + '\r')

    def write(self, text):
        with self.mutex:
            if text.strip():
                self._clear_progress()
            self.terminal.write(text)
            self.log.write(text)
            self.log.flush()
        return len(text)

    def write_progress(self, text):
        """Redraw the terminal, but persist occasional cursor-free snapshots."""
        return self._write_progress(text)

    def write_progress_snapshot(self, text, snapshot, *, source=None):
        """Keep the complete progress record independent of terminal width."""
        return self._write_progress(text, snapshot=snapshot, source=source)

    def _write_progress(self, text, snapshot=None, source=None):
        with self.mutex:
            self.terminal.write(text)
            state = self.progress_state
            clean = re.sub(r'\x1b\[[0-9;?]*[A-Za-z]', '', text).replace('\r', '').strip()
            if clean:
                state['pending'] = snapshot.strip() if snapshot is not None else clean
                if self.isatty():
                    state.update(terminal=self.terminal, width=len(clean))
            now = self.clock()
            final = '\n' in text
            changed_source = source is not None and source is not state.get('source')
            if source is not None:
                state['source'] = source
            pending = state.get('pending', '')
            if pending and (final or changed_source or now - state.get('logged_at', float('-inf')) >= 15):
                # tqdm.close can redraw an identical final line twice.
                if pending != state.get('logged'):
                    self.log.write(pending + '\n')
                    self.log.flush()
                    state.update(logged=pending, logged_at=now)
            if final:
                state.pop('terminal', None)
                state.pop('pending', None)
        return len(text)

    def flush(self):
        with self.mutex:
            self.terminal.flush()
            self.log.flush()

    def isatty(self):
        return bool(getattr(self.terminal, 'isatty', lambda: False)())

    def __getattr__(self, name):
        return getattr(self.terminal, name)


@contextmanager
def console_run(workspace):
    """Publish the whole main console, including preprocessing and terminal status."""
    owner = workspace_activity(workspace)["owner"]
    state = {"owner_token": owner.get("token"), "pid": owner.get("pid"), "status": "running",
             "started_utc": datetime.now(timezone.utc).isoformat(), "exit_code": None}
    with (workspace / CONSOLE).open("w", encoding="utf-8") as log:
        mutex = threading.RLock()
        progress_state = {}
        with redirect_stdout(Tee(sys.stdout, log, mutex, progress_state)), redirect_stderr(Tee(sys.stderr, log, mutex, progress_state)):
            write_provenance(workspace / STATE, state)
            print(f"[run] Console log: {workspace / CONSOLE}", flush=True)
            try:
                yield state
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    state["exit_code"] = 130
                elif isinstance(exc, SystemExit):
                    state["exit_code"] = (0 if exc.code is None else
                                          exc.code if isinstance(exc.code, int) else 1)
                else:
                    state["exit_code"] = 2
                traceback.print_exc()
                raise
            finally:
                if state["exit_code"] is None:
                    state["exit_code"] = 2
                state["status"] = ("completed" if state["exit_code"] == 0 else
                                   "interrupted" if state["exit_code"] in (130, 143) else "failed")
                state["finished_utc"] = datetime.now(timezone.utc).isoformat()
                print(f"[run] {state['status']}; exit code {state['exit_code']}", flush=True)
                write_provenance(workspace / STATE, state)


VIEW_RE = re.compile(r"Processing view\s+(\d+)\s*/\s*(\d+)(?: for (.+))?")
NATIVE_PREFIX_RE = re.compile(r"^([IWEF])\d{8}\s+\S+\s+\d+\s+[^\]]+\]\s*")
OUTPUT_RE = re.compile(r"Writing (photometric|geometric) output(?: for (.+))?")
COUNTED_RE = re.compile(r"^(Processed file|Undistorting image|Fusing image|Processing block|Processing batch|Matching image)\s*\[(\d+)/(\d+)\]")
REGISTER_RE = re.compile(r"Registering image #\d+ \(num_reg_(?:frames|images)=(\d+)\)")
STAGE_RE = re.compile(r"^\[(?:\d+[a-z]?/\d+|preflight|finalize|boundary|report)\]", re.I)
NUMBERED_STAGE_RE = re.compile(r"^\[\d+[a-z]?/\d+\]\s*(.+)", re.I)
NAMED_STAGES = {
    "preflight": "Preflight",
    "finalize": "Finalizing dense output...",
    "boundary": "Analyzing WiFi boundary hypotheses...",
    "report": "Generating reconstruction quality report...",
}
ALERT_RE = re.compile(r"\b(?:error|warning|fatal|failed|failure|traceback)\b", re.I)


def duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class StageTimings:
    """Account for all runner time, including finalization and reporting."""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.started = self.stage_started = clock()
        self.label = 'Preflight'
        self.stages = []

    def observe(self, line):
        text = line.strip()
        match = NUMBERED_STAGE_RE.match(text)
        label = match.group(1) if match else NAMED_STAGES.get(text.partition("]")[0].lstrip("[").lower())
        if label and label != self.label:
            now = self.clock()
            self._append(now)
            self.label = label
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
                'interpretation': 'Wall-clock time between stage markers, including subprocess startup and output handling.'}


class CompactProgress:
    """Render measured subtask progress, retaining raw solver output on disk."""

    def __init__(self, *, stream=None, clock=time.monotonic, estimate=True, sample_views=2, verbose=False, scope="colmap"):
        self.stream = stream if stream is not None else sys.stdout
        self.clock = clock
        self.estimate = estimate
        self.verbose = verbose
        self.sample_views = max(1, sample_views)
        self.scope = scope
        self.work = None
        self.kind = None
        self.pending_count = 0
        self.patch_completed = set()
        self.patch_names = set()
        self.patch_phase = ""
        self.patch_total = 0
        self.ba_count = 0
        self.mapper_backend = "CPU"
        self.feature_backend = "backend pending"
        self.matching_backend = "backend pending"
        self.view = self.total = 0
        self.last_message = ""
        self.pending_severity = None
        self.last_console_output = float("-inf")
        self.repeated_warning = None
        self.warning_repeats = 0
        self.warning_reported_at = 0.0

    def clear(self):
        if self.work is not None and self.work.bar is not None:
            self.work.bar.clear()

    def message(self, text):
        self.clear()
        tqdm.write(text, file=self.stream)
        self.stream.flush()
        self.last_console_output = self.clock()

    def flush_warning_repeats(self):
        if self.warning_repeats:
            self.message(f"[warn] Previous warning repeated {self.warning_repeats} more times; see full log.")
            self.warning_repeats = 0
        self.warning_reported_at = self.clock()

    def progress(self):
        if self.work is not None and not self.verbose:
            self.work.refresh()
            self.last_console_output = max(self.last_console_output, self.work.last_render)

    def start_work(self, kind, label, total=None, unit="items"):
        self.finish_pass()
        self.kind = kind
        if not self.verbose:
            self.work = WorkProgress(label, total, unit, stream=self.stream, clock=self.clock,
                                     min_samples=self.sample_views, estimate=self.estimate)
            self.last_console_output = self.clock()

    def count(self, completed):
        if self.work is not None:
            self.work.update(max(self.work.completed, completed))
            self.last_console_output = max(self.last_console_output, self.work.last_render)

    def finish_pass(self, status=None):
        if self.work is not None:
            self.work.close(status)
            self.work = None
            self.last_console_output = self.clock()
        self.kind = None
        self.pending_count = 0
        self.view = self.total = 0

    def close(self, status=None):
        self.flush_warning_repeats()
        self.finish_pass(status)

    def observe_counted(self, text):
        match = COUNTED_RE.match(text)
        if match:
            kind, index, total = match.groups()
            index, total = int(index), int(total)
            if not 1 <= index <= total:
                return True
            labels = {"Processed file": f"SIFT feature extraction [{self.feature_backend}]",
                      "Undistorting image": "Image undistortion [CPU]",
                      "Fusing image": "Stereo fusion [CPU]",
                      "Processing block": f"Feature matching blocks [{self.matching_backend} + CPU verification]",
                      "Processing batch": f"Feature matching batches [{self.matching_backend} + CPU verification]",
                      "Matching image": f"Feature matching [{self.matching_backend} + CPU verification]"}
            if kind != self.kind or (self.work is not None and self.work.total != total):
                self.start_work(kind, labels[kind], total,
                                "blocks" if "block" in kind else "images")
            self.pending_count = index
            # COLMAP logs undistortion/fusion/matching *before* doing the work.
            self.count(index if kind == "Processed file" else index - 1)
            return True
        if self.kind in {"Processing block", "Processing batch", "Matching image", "Fusing image"} and re.match(r"^in \d+(?:\.\d+)?s\b", text):
            self.count(self.pending_count)
            return True
        if self.kind == "Undistorting image" and text == "Writing reconstruction...":
            self.count(self.pending_count)
            self.finish_pass()
            return True
        if text.startswith("Elapsed time:") and self.kind and self.kind != "mapper":
            self.finish_pass("finished")
            return True
        registered = REGISTER_RE.search(text)
        if registered or "Global bundle adjustment" in text:
            if self.kind != "mapper":
                self.start_work("mapper", "Sparse mapping", unit="registered frames")
            if registered:
                # The count describes frames already registered, not success
                # of the new attempted image; failures and restarts are possible.
                n = int(registered.group(1))
                if self.work is not None and n < self.work.completed:
                    self.start_work("mapper", "Sparse mapping: new model", unit="registered frames")
                if self.work is not None:
                    self.work.label = "Sparse mapping: registering cameras"
                self.count(n)
            else:
                self.ba_count += 1
                if self.work is not None:
                    self.work.label = f"Sparse global BA #{self.ba_count} [{self.mapper_backend}]"
                self.progress()
            return True
        if "Finding good initial image pair" in text:
            self.start_work("mapper", "Sparse mapping: searching initial pair", unit="registered frames")
            return True
        return False

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
        # Child Python stages use the same bounded-work renderer. Its status
        # text is not a native warning and must not be parsed a second time.
        if text.startswith('[progress]'):
            if not self.verbose:
                self.finish_pass()
                self.message(text)
            return
        if (severity in {"W", "E", "F"} or ALERT_RE.search(text) or
                text.lower().startswith(("[warn]", "[warning]"))):
            if self.verbose:
                return  # The caller already printed the complete native line.
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
        if self.verbose:
            return
        backend = re.search(r'Creating SIFT (GPU|CPU) feature (extractor|matcher)', text)
        if backend:
            if backend.group(2) == 'extractor':
                self.feature_backend = backend.group(1)
            else:
                self.matching_backend = backend.group(1)
            return
        if self.observe_counted(text):
            return
        if text.startswith('[resources]') and 'GPU BA=' in text:
            self.mapper_backend = 'GPU + CPU' if re.search(r'GPU BA=(?:True|enabled)', text) else 'CPU'
        view = VIEW_RE.search(text)
        if view:
            index, total = map(int, view.groups()[:2])
            name = view.group(3) or str(index)
            if not 1 <= index <= total:
                return
            if self.kind != 'patch' or total != self.patch_total:
                self.start_work('patch', 'PatchMatch [GPU; current pass]', total, 'views')
                self.patch_phase = ''
                self.patch_completed.clear()
                self.patch_names.clear()
                if index > 1:
                    self.message('[estimate] Cached views may be skipped; PatchMatch ETA is an upper bound for this pass.')
            self.patch_total = total
            self.patch_names.add(name)
            self.view, self.total = index, total
            # Starting an indexed view is not completion of earlier views:
            # multiple GPU workers can start/finish out of order.
            self.progress()
            return
        option = re.match(r'geom_consistency:\s*([01])\s*$', text)
        if option and self.kind == 'patch':
            self.set_patch_phase('geometric' if option.group(1) == '1' else 'photometric')
            return
        output = OUTPUT_RE.search(text)
        if output and self.kind == 'patch':
            self.set_patch_phase(output.group(1))
            name = output.group(2) or str(self.view)
            self.patch_completed.add(name)
            self.count(len(self.patch_completed))
            return
        if text.startswith('{') and '"depth_reference_count"' in text:
            try:
                plan = json.loads(text)
                count, total = plan['depth_reference_count'], plan['view_count']
                if (type(count) is int and type(total) is int and 0 <= count <= total):
                    self.message(f"[depth] {count}/{total} registered views selected as depth references; "
                                 f"{total - count} views not selected for depth estimation.")
                    return
            except (ValueError, KeyError, TypeError):
                pass
        if text.lower().startswith(("[coverage]", "[provenance]", "[load]",
                                    "[snapshot]", "[sparse-log]", "[preflight]")):
            return  # Routine metadata remains available in the full log.
        if STAGE_RE.match(text):
            self.flush_warning_repeats()
            self.repeated_warning = None
            self.finish_pass('finished')
            if text != self.last_message:
                self.message(f'[stage] {self.scope} {text}')
                self.last_message = text
        elif text.startswith("[") and text != self.last_message:
            self.message(text)
            self.last_message = text
        # Native option dumps, timestamps and separators remain in the log only.
        else:
            self.progress()

    def set_patch_phase(self, phase):
        if self.patch_phase and phase != self.patch_phase:
            total, index = self.patch_total, self.view
            self.start_work('patch', f'PatchMatch {phase} [GPU; current pass]', total, 'views')
            self.view, self.total = index, total
            self.patch_completed.clear()
            self.patch_names.clear()
        self.patch_phase = phase
        if self.work is not None:
            self.work.label = f'PatchMatch {phase} [GPU; current pass]'


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


def stream_command(command, *, workspace, env, verbose=False,
                   estimate=True, sample_views=2, label="colmap",
                   heartbeat_seconds=60, output_timeout=None) -> int:
    """Drain stdout/stderr continuously; retain details even on failure or Ctrl-C."""
    log_dir = Path(workspace) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    env = dict(env)
    # Keep Python stage output and this pipe decoder in agreement independently
    # of the Windows ANSI locale. Native tool diagnostics remain best-effort.
    env["PYTHONIOENCODING"] = "utf-8"
    env["FYP_PROGRESS_ESTIMATE"] = '1' if estimate else '0'
    env["FYP_COLMAP_RUN_ID"] = stamp
    if output_timeout is None:
        output_timeout = float(env.get("COLMAP_OUTPUT_TIMEOUT_SECONDS", "0"))
    if not math.isfinite(output_timeout) or output_timeout < 0:
        raise ValueError("COLMAP_OUTPUT_TIMEOUT_SECONDS must be finite and nonnegative")
    log_path = log_dir / f"{label}_{stamp}.log"
    env["FYP_COLMAP_LOG_PATH"] = str(log_path.resolve())
    scope = 'dense' if 'dense' in label else 'sparse' if 'sparse' in label else label
    display = CompactProgress(estimate=estimate, sample_views=sample_views, verbose=verbose, scope=scope)
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
                                 text=True, encoding="utf-8", errors="replace") as process:
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
                            # The sparse receipt seals only a prefix containing
                            # this end-of-mapper marker. Publish all preceding
                            # solver output before its child reads that prefix.
                            if line.startswith("[sparse-log]") or STAGE_RE.match(line):
                                log.flush()
                            tail.append(line.rstrip())
                            if verbose:
                                print(line, end="", flush=True)
                            display.observe(line)
                        display.progress()
                        if now - last_heartbeat >= heartbeat_seconds:
                            message = (f"[monitor] PID {process.pid} | {timings.label} | "
                                       f"stage elapsed {duration(now - timings.stage_started)} | "
                                       f"run elapsed {duration(now - started)} | ETA unavailable | "
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
        display.close('failed' if code else 'finished')
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
        display.close('interrupted' if status == 'interrupted_or_failed' else status)
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
