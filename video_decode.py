"""Prefer available video hardware without reducing the CPU worker budget."""
import os
from pathlib import Path

from compute_resources import worker_count


def windows_nvidia_adapter():
    """Return the DXGI adapter ordinal, which is not the CUDA device ordinal."""
    if os.name != 'nt':
        return None
    import ctypes as ct
    import uuid
    class Description(ct.Structure):
        _fields_ = [('name', ct.c_wchar * 128), ('vendor', ct.c_uint),
                    ('device', ct.c_uint), ('subsystem', ct.c_uint), ('revision', ct.c_uint),
                    ('video_memory', ct.c_size_t), ('system_memory', ct.c_size_t),
                    ('shared_memory', ct.c_size_t), ('luid_low', ct.c_uint), ('luid_high', ct.c_long)]
    def method(instance, index, result, *arguments):
        address = ct.cast(instance, ct.POINTER(ct.POINTER(ct.c_void_p))).contents[index]
        return ct.WINFUNCTYPE(result, ct.c_void_p, *arguments)(address)
    factory = ct.c_void_p()
    try:
        create = ct.WinDLL('dxgi').CreateDXGIFactory
        create.argtypes = [ct.c_void_p, ct.POINTER(ct.c_void_p)]
        create.restype = ct.c_long
        iid = (ct.c_ubyte * 16).from_buffer_copy(uuid.UUID('7b7166ec-21c7-44ae-b21a-c9ae321ae369').bytes_le)
        if create(ct.byref(iid), ct.byref(factory)) != 0:
            return None
        for index in range(32):
            adapter = ct.c_void_p()
            if method(factory, 7, ct.c_long, ct.c_uint, ct.POINTER(ct.c_void_p))(factory, index, ct.byref(adapter)) != 0:
                break
            try:
                description = Description()
                if method(adapter, 8, ct.c_long, ct.POINTER(Description))(adapter, ct.byref(description)) == 0 and description.vendor == 0x10DE:
                    return index, description.name
            finally:
                method(adapter, 2, ct.c_ulong)(adapter)
    except (OSError, AttributeError):
        return None
    finally:
        if factory:
            method(factory, 2, ct.c_ulong)(factory)
    return None


def decoder_info(capture, hardware='off', preferred=None):
    import cv2
    mode = int(capture.get(cv2.CAP_PROP_HW_ACCELERATION)) if hasattr(cv2, 'CAP_PROP_HW_ACCELERATION') else 0
    device = int(capture.get(cv2.CAP_PROP_HW_DEVICE)) if mode else -1
    names = {0: 'software', getattr(cv2, 'VIDEO_ACCELERATION_D3D11', 2): 'd3d11',
             getattr(cv2, 'VIDEO_ACCELERATION_VAAPI', 3): 'vaapi', getattr(cv2, 'VIDEO_ACCELERATION_MFX', 4): 'mfx'}
    return {'requested_hardware': hardware, 'actual_backend': names.get(mode, f'unknown_{mode}'),
            'hardware_active': mode in names and mode != 0, 'device_index': device,
            'device_name': preferred[1] if preferred and device == preferred[0] and mode == getattr(cv2, 'VIDEO_ACCELERATION_D3D11', 2) else None}


def open_video(path, threads=0, *, report=True, hardware='auto', device=-1, diagnostics=None):
    import cv2

    if hardware not in ('auto', 'off') or type(device) is not int or device < -1:
        raise ValueError('Video decoder hardware must be auto/off and device must be -1 or nonnegative')
    budget = worker_count(threads)
    preferred = windows_nvidia_adapter() if hardware == 'auto' and device == -1 else None
    thread_options = [cv2.CAP_PROP_N_THREADS, budget] if hasattr(cv2, 'CAP_PROP_N_THREADS') else []
    if hardware == 'auto' and hasattr(cv2, 'CAP_PROP_HW_ACCELERATION'):
        selected_device = preferred[0] if preferred else device
        mode = (cv2.VIDEO_ACCELERATION_D3D11 if os.name == 'nt' else cv2.VIDEO_ACCELERATION_VAAPI) if selected_device >= 0 else cv2.VIDEO_ACCELERATION_ANY
        options = [*thread_options, cv2.CAP_PROP_HW_ACCELERATION, mode]
        if selected_device >= 0:
            options += [cv2.CAP_PROP_HW_DEVICE, selected_device]
        accelerated = None
        try:
            accelerated = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, options)
            if accelerated.isOpened():
                info = decoder_info(accelerated, hardware, preferred)
                if diagnostics is not None:
                    diagnostics.update(info)
                if report:
                    print(f"[resources] {Path(path).name}: decoder={info['actual_backend']}, "
                          f"device={info['device_name'] or info['device_index']}; CPU thread budget={budget}.", flush=True)
                return accelerated
        except cv2.error:
            pass
        if accelerated is not None:
            accelerated.release()
        if report:
            print(f'[resources] {Path(path).name}: hardware decoder unavailable; retaining CPU decoding with {budget} threads.', flush=True)
    capture = None
    if hasattr(cv2, 'CAP_PROP_N_THREADS') and hasattr(cv2, 'CAP_FFMPEG'):
        try:
            # This is an open-only option. Calling set() after VideoCapture()
            # leaves OpenCV 4.10's default FFmpeg thread limit in effect.
            capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG,
                                       [cv2.CAP_PROP_N_THREADS, budget])
            if capture.isOpened():
                if report:
                    actual = int(capture.get(cv2.CAP_PROP_N_THREADS))
                    print(f'[resources] {Path(path).name}: FFmpeg decoder threads={actual or "unreported"} '
                          f'(requested {budget}).', flush=True)
                if diagnostics is not None:
                    diagnostics.update(decoder_info(capture, hardware))
                return capture
        except cv2.error:
            pass
        if capture is not None:
            capture.release()
    capture = cv2.VideoCapture(str(path))
    if report and capture.isOpened():
        print(f'[warn] {Path(path).name}: decoder thread option unavailable; '
              'using backend defaults.', flush=True)
    if diagnostics is not None and capture.isOpened():
        diagnostics.update(decoder_info(capture, hardware))
    return capture
