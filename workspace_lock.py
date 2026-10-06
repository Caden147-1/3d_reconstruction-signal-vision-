"""Nonblocking OS workspace locks, shared only with explicitly delegated children."""

import argparse
from contextlib import contextmanager
from functools import wraps
import hashlib
import inspect
import json
import os
from pathlib import Path
import threading
import uuid


class WorkspaceBusyError(RuntimeError):
    pass


_owners = {}
_mutex = threading.RLock()


def lock_key(workspace):
    return os.path.normcase(str(Path(workspace).resolve()))


def token_variable(key):
    return "FYP_LOCK_" + hashlib.sha256(key.encode()).hexdigest()[:24].upper()


def kernel_lock(handle, unlock=False):
    handle.seek(0)
    if os.name == "nt":
        import msvcrt
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN if unlock else fcntl.LOCK_EX | fcntl.LOCK_NB)


def delegated(handle, key):
    token = os.environ.get(token_variable(key))
    if not token:
        return False
    try:
        # Byte zero belongs to the OS lock; Windows denies reading it in children.
        handle.seek(1)
        owner = json.loads(handle.read().decode("utf-8"))
        return owner.get("token") == token and owner.get("workspace") == key
    except (OSError, ValueError, UnicodeError):
        return False


def inherited_lock_held(workspace):
    key = lock_key(workspace)
    path = Path(key) / ".pipeline.lock"
    try:
        with path.open("r+b", buffering=0) as handle:
            try:
                kernel_lock(handle)
            except OSError:
                return delegated(handle, key)
            kernel_lock(handle, unlock=True)
    except OSError:
        pass
    return False


def workspace_activity(workspace):
    """Probe the OS lock without changing its owner record or trusting a stale PID."""
    key = lock_key(workspace)
    path = Path(key) / ".pipeline.lock"
    try:
        with path.open("r+b", buffering=0) as handle:
            with _mutex:
                active = key in _owners
            if not active:
                try:
                    kernel_lock(handle)
                except OSError:
                    active = True
                else:
                    kernel_lock(handle, unlock=True)
            handle.seek(1)
            try:
                owner = json.loads(handle.read().decode("utf-8"))
            except (ValueError, UnicodeError):
                owner = {}
            return {"active": active, "owner": owner}
    except FileNotFoundError:
        return {"active": False, "owner": {}}


@contextmanager
def workspace_lock(workspace):
    key = lock_key(workspace)
    thread = threading.get_ident()
    handle = None
    acquired = False
    nested = False
    previous_token = None
    variable = token_variable(key)
    with _mutex:
        if key in _owners:
            if _owners[key][0] != thread:
                raise WorkspaceBusyError(f"Workspace is already in use: {key}")
            _owners[key][1] += 1
            nested = True
        else:
            Path(key).mkdir(parents=True, exist_ok=True)
            fd = os.open(Path(key) / ".pipeline.lock", os.O_RDWR | os.O_CREAT, 0o600)
            handle = os.fdopen(fd, "r+b", buffering=0)
            try:
                kernel_lock(handle)
                acquired = True
                token = uuid.uuid4().hex
                owner = {"workspace": key, "pid": os.getpid(), "token": token}
                handle.seek(0)
                handle.write(b" " + json.dumps(owner).encode("utf-8"))
                handle.truncate()
                previous_token = os.environ.get(variable)
                os.environ[variable] = token
            except OSError as exc:
                if acquired or not delegated(handle, key):
                    if acquired:
                        kernel_lock(handle, unlock=True)
                    handle.close()
                    raise WorkspaceBusyError(f"Cannot exclusively access workspace: {key}") from exc
            _owners[key] = [thread, 1]
    try:
        yield
    finally:
        with _mutex:
            _owners[key][1] -= 1
            if not _owners[key][1]:
                del _owners[key]
            if not nested:
                if acquired:
                    if previous_token is None:
                        os.environ.pop(variable, None)
                    else:
                        os.environ[variable] = previous_token
                    kernel_lock(handle, unlock=True)
                handle.close()
                # Never unlink the lock file: waiters must lock the same inode.


def locked_workspace(parameter):
    def decorate(function):
        signature = inspect.signature(function)

        @wraps(function)
        def wrapped(*args, **kwargs):
            workspace = signature.bind(*args, **kwargs).arguments[parameter]
            with workspace_lock(workspace):
                return function(*args, **kwargs)
        return wrapped
    return decorate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--check-owned", action="store_true", required=True)
    args = parser.parse_args()
    raise SystemExit(0 if inherited_lock_held(args.workspace) else 1)
