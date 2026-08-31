import os
import json
import tempfile
from pathlib import Path
from tqdm.auto import tqdm
import time

def save_progress(path, state):
    """
    Atomically write a checkpoint dictionary to a JSON file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.stem + "_", suffix=".tmp")
    with os.fdopen(fd, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, path)

def load_progress(path):
    """
    Load a checkpoint dictionary from a JSON file.
    Returns an empty dict if the file does not exist.
    """
    path = Path(path)
    if path.exists():
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def append_result(path, row):
    """
    Append one JSON-line result to a .jsonl file, flushing immediately.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(row) + '\n')
        f.flush()
        os.fsync(f.fileno())

def load_completed_keys(path, key_field='key'):
    """
    Read a .jsonl results file and return the set of already-completed keys.
    """
    path = Path(path)
    if not path.exists():
        return set()
    keys = set()
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if key_field in data:
                    keys.add(data[key_field])
            except json.JSONDecodeError:
                continue
    return keys

def run_resumable(items, key_fn, process_fn, results_path,
                  max_runtime_seconds=None, desc='Processing', key_field='key'):
    """
    Generic resumable loop with checkpointing and tqdm progress bar.
    """
    results_path = Path(results_path)
    completed_keys = load_completed_keys(results_path, key_field=key_field)
    remaining_items = [item for item in items if key_fn(item) not in completed_keys]

    print(f"Resumable run: {len(completed_keys)} already completed, {len(remaining_items)} remaining.")

    if not remaining_items:
        print("All items already processed. Nothing to do.")
        return

    start_time = time.time()
    with tqdm(total=len(items), initial=len(completed_keys), desc=desc) as pbar:
        for item in remaining_items:
            key = key_fn(item)
            result = process_fn(item)

            if isinstance(result, dict):
                result[key_field] = key
            else:
                result = {key_field: key, 'result': result}

            append_result(results_path, result)
            pbar.update(1)

            if max_runtime_seconds is not None:
                elapsed = time.time() - start_time
                if elapsed >= max_runtime_seconds:
                    remaining_after_stop = len(remaining_items) - (pbar.n - len(completed_keys))
                    print(f"Runtime budget {max_runtime_seconds}s exceeded. "
                          f"Stopping gracefully after {pbar.n - len(completed_keys)} items. "
                          f"{remaining_after_stop} items remain for the next run.")
                    break
