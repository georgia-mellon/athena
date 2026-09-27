import os
import sys

if sys.platform == "darwin" and not os.environ.get("ATHENA_OMP_FIXED"):
    # ponytail: torch bundles its own libomp and lightgbm links Homebrew's; two OpenMP runtimes in one process segfault
    # Hearsay E5 (XLS-R + LightGBM) on macOS. DYLD_LIBRARY_PATH makes lightgbm resolve torch's copy, but dyld only
    # reads it at process start, so re-exec once with it set. Upgrade path: a lightgbm build that uses torch's libomp.
    import importlib.util
    spec = importlib.util.find_spec("torch")
    if spec and spec.origin and getattr(sys, "orig_argv", None):
        lib = os.path.join(os.path.dirname(spec.origin), "lib")
        env = {**os.environ, "ATHENA_OMP_FIXED": "1",
               "DYLD_LIBRARY_PATH": os.pathsep.join(filter(None, [lib, os.environ.get("DYLD_LIBRARY_PATH")]))}
        os.execve(sys.executable, sys.orig_argv, env)
