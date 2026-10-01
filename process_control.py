"""Own subprocess trees until exit, including errors and Windows parent death."""

from contextlib import contextmanager
import os
import signal
import subprocess
import threading
import time


class WindowsJob:
    def __init__(self):
        import ctypes as c
        from ctypes import wintypes as w

        class Limits(c.Structure):
            _fields_ = [("user", c.c_longlong), ("job_user", c.c_longlong),
                        ("flags", w.DWORD), ("min_ws", c.c_size_t), ("max_ws", c.c_size_t),
                        ("process_limit", w.DWORD), ("affinity", c.c_size_t),
                        ("priority", w.DWORD), ("scheduling", w.DWORD)]

        class ExtendedLimits(c.Structure):
            _fields_ = [("basic", Limits), ("io", c.c_ulonglong * 6),
                        ("process_memory", c.c_size_t), ("job_memory", c.c_size_t),
                        ("peak_process", c.c_size_t), ("peak_job", c.c_size_t)]

        class Accounting(c.Structure):
            _fields_ = [("times", c.c_longlong * 4), ("faults", w.DWORD),
                        ("total", w.DWORD), ("active", w.DWORD), ("terminated", w.DWORD)]

        class ThreadEntry(c.Structure):
            _fields_ = [("size", w.DWORD), ("usage", w.DWORD), ("tid", w.DWORD),
                        ("pid", w.DWORD), ("base_priority", w.LONG),
                        ("delta_priority", w.LONG), ("flags", w.DWORD)]

        self.c, self.Accounting, self.ThreadEntry = c, Accounting, ThreadEntry
        self.api = c.WinDLL("kernel32", use_last_error=True)
        signatures = {
            "CreateJobObjectW": ([c.c_void_p, w.LPCWSTR], w.HANDLE),
            "SetInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD], w.BOOL),
            "QueryInformationJobObject": ([w.HANDLE, c.c_int, c.c_void_p, w.DWORD, c.c_void_p], w.BOOL),
            "AssignProcessToJobObject": ([w.HANDLE, w.HANDLE], w.BOOL),
            "TerminateJobObject": ([w.HANDLE, w.UINT], w.BOOL),
            "CloseHandle": ([w.HANDLE], w.BOOL),
            "CreateToolhelp32Snapshot": ([w.DWORD, w.DWORD], w.HANDLE),
            "Thread32First": ([w.HANDLE, c.POINTER(ThreadEntry)], w.BOOL),
            "Thread32Next": ([w.HANDLE, c.POINTER(ThreadEntry)], w.BOOL),
            "OpenThread": ([w.DWORD, w.BOOL, w.DWORD], w.HANDLE),
            "ResumeThread": ([w.HANDLE], w.DWORD),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.api, name)
            function.argtypes, function.restype = args, result
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise c.WinError(c.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, c.byref(limits), c.sizeof(limits)):
            error = c.WinError(c.get_last_error())
            self.api.CloseHandle(self.handle)
            raise error

    def attach_and_resume(self, process):
        c, api = self.c, self.api
        if not api.AssignProcessToJobObject(self.handle, int(process._handle)):
            raise c.WinError(c.get_last_error())
        snapshot = api.CreateToolhelp32Snapshot(0x4, 0)  # TH32CS_SNAPTHREAD
        if snapshot == c.c_void_p(-1).value:
            raise c.WinError(c.get_last_error())
        try:
            entry = self.ThreadEntry()
            entry.size = c.sizeof(entry)
            found = api.Thread32First(snapshot, c.byref(entry))
            while found:
                if entry.pid == process.pid:
                    thread = api.OpenThread(0x2, False, entry.tid)  # THREAD_SUSPEND_RESUME
                    if not thread:
                        raise c.WinError(c.get_last_error())
                    try:
                        if api.ResumeThread(thread) == 0xFFFFFFFF:
                            raise c.WinError(c.get_last_error())
                    finally:
                        api.CloseHandle(thread)
                    return
                entry.size = c.sizeof(entry)
                found = api.Thread32Next(snapshot, c.byref(entry))
            raise OSError("Cannot find the suspended child thread")
        finally:
            api.CloseHandle(snapshot)

    def close(self):
        c, api = self.c, self.api
        try:
            if not api.TerminateJobObject(self.handle, 1):
                raise c.WinError(c.get_last_error())
            # Do not release the workspace lock while descendants still exist.
            while True:
                info = self.Accounting()
                if not api.QueryInformationJobObject(self.handle, 1, c.byref(info), c.sizeof(info), None):
                    raise c.WinError(c.get_last_error())
                if not info.active:
                    break
                time.sleep(.02)
        finally:
            api.CloseHandle(self.handle)

    def active_processes(self):
        info = self.Accounting()
        if not self.api.QueryInformationJobObject(
                self.handle, 1, self.c.byref(info), self.c.sizeof(info), None):
            raise self.c.WinError(self.c.get_last_error())
        return info.active


@contextmanager
def managed_process(command, **kwargs):
    job = WindowsJob() if os.name == "nt" else None
    process = None
    try:
        if job:
            # Attach before user code can spawn descendants outside the job.
            kwargs["creationflags"] = kwargs.get("creationflags", 0) | 0x4  # CREATE_SUSPENDED
        else:
            kwargs["start_new_session"] = True
        process = subprocess.Popen(command, **kwargs)
        if job:
            job.attach_and_resume(process)
            process.active_tree_processes = job.active_processes
        yield process
    finally:
        main_thread = threading.current_thread() is threading.main_thread()
        previous = signal.signal(signal.SIGINT, signal.SIG_IGN) if main_thread else None
        try:
            try:
                if job:
                    job.close()
                elif process:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
            finally:
                if process:
                    if process.poll() is None:
                        process.kill()
                    process.wait()
                    for stream in (process.stdin, process.stdout, process.stderr):
                        if stream:
                            stream.close()
                    if os.name == "nt":
                        process._handle.Close()
        finally:
            if main_thread:
                signal.signal(signal.SIGINT, previous)
