#!/usr/bin/env python3
"""Build the Seera C++ engine shared library."""
import subprocess
import sysconfig
import sys
import os

import numpy as np

def build():
    root = os.path.dirname(os.path.abspath(__file__))
    engine_dir = os.path.join(root, "engine")

    python_include = sysconfig.get_config_var("INCLUDEPY")
    numpy_include  = np.get_include()
    ext_suffix     = sysconfig.get_config_var("EXT_SUFFIX")
    output         = os.path.join(root, f"seera_cpp{ext_suffix}")

    src_files = [
        os.path.join(engine_dir, "src", f)
        for f in [
            "tensor_ops.cpp",
            "activation_ops.cpp",
            "conv_ops.cpp",
            "pool_ops.cpp",
            "batchnorm_ops.cpp",
            "bindings.cpp",
        ]
    ]

    # Get pybind11 include path from the pip package (system version too old for Python 3.13)
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError:
        print("✗ pybind11 not installed. Run: pip install pybind11")
        sys.exit(1)

    cmd = [
        "g++",
        "-O3",                        # max optimization
        "-shared",                    # shared library
        "-fPIC",                      # position independent code
        "-fopenmp",                   # OpenMP
        "-std=c++17",                 # C++17
        f"-I{os.path.join(engine_dir, 'include')}",
        f"-I{python_include}",
        f"-I{numpy_include}",
        f"-I{pybind11_include}",      # pybind11 from pip
    ] + src_files + [
        "-L/usr/lib/x86_64-linux-gnu/openblas-pthread/",
        "-lopenblas",
        "-o", output,
    ]

    print("Building Seera C++ engine...")
    print(f"  Output: {output}")
    print(f"  Command: {' '.join(cmd[:8])} ...")

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n✗ Build FAILED:\n{result.stderr}")
        sys.exit(1)

    print(f"✓ Build successful: {output}")
    print(f"  Size: {os.path.getsize(output) / 1024:.0f} KB")

    # Quick import test
    sys.path.insert(0, root)
    try:
        import importlib
        mod = importlib.import_module("seera_cpp")
        print(f"✓ Import test passed: {mod.__doc__}")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    build()
