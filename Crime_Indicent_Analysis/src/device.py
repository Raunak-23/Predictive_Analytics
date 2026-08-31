import importlib
import sys

def get_device_info():
    """
    Detect available hardware acceleration libraries.
    Returns a dict with boolean flags for torch_cuda, cudf, and cuml,
    plus the GPU name if torch.cuda is available.
    """
    info = {'torch_cuda': False, 'gpu_name': None, 'cudf': False, 'cuml': False}

    # Torch CUDA
    try:
        import torch
        if torch.cuda.is_available():
            info['torch_cuda'] = True
            info['gpu_name'] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # RAPIDS cuDF
    try:
        import cudf
        info['cudf'] = True
    except ImportError:
        pass

    # RAPIDS cuML
    try:
        import cuml
        info['cuml'] = True
    except ImportError:
        pass

    return info
