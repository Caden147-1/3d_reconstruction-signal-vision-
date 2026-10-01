"""Run independent sensor branches in owned processes, then join before fusion."""

from contextlib import ExitStack
from collections import deque
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from process_control import managed_process


def sensor_stage(stage, payload):
    from preprocess_pipeline import build_pdr_tables, build_wifi_tables
    from trajectory_contour import build_trajectory_contour

    sessions = [Path(path) for path in payload['sessions']]
    output = Path(payload['output'])
    if stage == 'wifi':
        return build_wifi_tables(sessions, output, **payload['wifi'])
    if stage == 'pdr':
        pdr = build_pdr_tables(sessions, output, **payload['pdr'])
        contour = build_trajectory_contour(
            input_path=output / 'pdr_path.csv',
            contour_path=output / 'pdr_trajectory_contour.csv',
            edges_path=output / 'pdr_trajectory_edges.csv',
            summary_path=output / 'pdr_trajectory_contour_summary.json')
        return {'pdr': pdr, 'trajectory_contour': contour}
    raise ValueError(f'Unknown sensor stage: {stage}')


def run_sensor_stages(session_dirs, output_dir, *, wifi_options, pdr_options, workers=2):
    """Caller holds its workspace lock until both writers have exited.

    The branches write disjoint files. Fusion starts only after both succeed.
    On failure or Ctrl-C, owned process trees exit before this function returns.
    """
    if workers not in (1, 2):
        raise ValueError('Sensor workers must be 1 or 2')
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload = {'sessions': [str(Path(path).resolve()) for path in session_dirs],
               'output': str(output), 'wifi': wifi_options, 'pdr': pdr_options}
    started = time.monotonic()
    results, timings = {}, {}
    if workers == 1:
        for stage in ('wifi', 'pdr'):
            stage_started = time.monotonic()
            results[stage] = sensor_stage(stage, payload)
            timings[stage] = {'start_seconds': stage_started - started,
                              'elapsed_seconds': time.monotonic() - stage_started}
    else:
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        log_dir = output / 'logs'
        log_dir.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='.sensor_workers_', dir=output) as folder:
            task_dir = Path(folder)
            spec = task_dir / 'inputs.json'
            spec.write_text(json.dumps(payload), encoding='utf-8')
            with ExitStack() as stack:
                pending = {}
                for stage in ('wifi', 'pdr'):
                    result_path = task_dir / f'{stage}.json'
                    log_path = log_dir / f'sensor_{stage}_{stamp}.log'
                    log = stack.enter_context(log_path.open('wb'))
                    command = [sys.executable, '-u', str(Path(__file__).resolve()),
                               stage, str(spec), str(result_path)]
                    stage_started = time.monotonic()
                    process = stack.enter_context(managed_process(
                        command, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT))
                    pending[stage] = (process, result_path, log_path, stage_started)
                last_update = time.monotonic()
                while pending:
                    for stage, (process, result_path, log_path, stage_started) in list(pending.items()):
                        code = process.poll()
                        if code is None:
                            continue
                        if code:
                            with log_path.open(encoding='utf-8', errors='replace') as diagnostic:
                                detail = ''.join(deque(diagnostic, maxlen=12)).strip()
                            raise ValueError(f'{stage} preprocessing exited with code {code}; '
                                             f'log: {log_path}\n{detail}')
                        results[stage] = json.loads(result_path.read_text(encoding='utf-8'))
                        timings[stage] = {'start_seconds': stage_started - started,
                                          'elapsed_seconds': time.monotonic() - stage_started}
                        if log_path.stat().st_size:
                            print(f'[warn] {stage} preprocessing emitted diagnostics: {log_path}', flush=True)
                        del pending[stage]
                    if pending:
                        if time.monotonic() - last_update >= 60:
                            print(f'[stage] Sensor preprocessing running: {", ".join(pending)}', flush=True)
                            last_update = time.monotonic()
                        time.sleep(0.1)
    results['execution'] = {'workers': workers, 'elapsed_seconds': time.monotonic() - started,
                            'stages': timings}
    return results


if __name__ == '__main__':
    stage, spec_path, result_path = sys.argv[1:]
    payload = json.loads(Path(spec_path).read_text(encoding='utf-8'))
    result = sensor_stage(stage, payload)
    Path(result_path).write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
