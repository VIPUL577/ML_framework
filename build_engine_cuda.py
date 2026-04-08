#!/usr/bin/env python3
"""Build the Seera CUDA engine shared library (seera_cuda)."""
import subprocess
import sysconfig
import sys
import os
import shutil

import numpy as np


def build():
    root = os.path.dirname(os.path.abspath(__file__))
    engine_dir = os.path.join(root, "engine_cuda")
    include_dir = os.path.join(engine_dir, "include")
    src_dir = os.path.join(engine_dir, "src")

    python_include = sysconfig.get_config_var("INCLUDEPY")
    numpy_include = np.get_include()
    ext_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    output = os.path.join(root, f"seera_cuda{ext_suffix}")

    # ── Gather source files ──────────────────────────────────────────
    cu_files = [
        os.path.join(src_dir, f)
        for f in [
            "elemops.cu",
            "activations.cu",
            "GEMM.cu",
            "convolution.cu",
            "col2im.cu",
            "maxPool.cu",
            "unpooling.cu",
            "upsampling.cu",
            "reductionKernels.cu",
            "cuTen_essentails.cu",
            "broadcast.cu",
            # "data_transfer.cu",
        ]
    ]
    bindings_cpp = os.path.join(src_dir, "cuda_bindings.cpp")

    # ── Resolve pybind11 include ─────────────────────────────────────
    try:
        import pybind11
        pybind11_include = pybind11.get_include()
    except ImportError:
        print("✗ pybind11 not installed. Run: pip install pybind11")
        sys.exit(1)

    # ── Detect CUDA toolkit ──────────────────────────────────────────
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        print("✗ nvcc not found. Make sure the CUDA toolkit is installed and on PATH.")
        sys.exit(1)

    # Infer CUDA home from nvcc location  (e.g. /usr/local/cuda/bin/nvcc)
    cuda_bin = os.path.dirname(os.path.realpath(nvcc))
    cuda_home = os.path.dirname(cuda_bin)
    cuda_include = os.path.join(cuda_home, "include")
    cuda_lib = os.path.join(cuda_home, "lib64")

    # ── Temp dir for object files ────────────────────────────────────
    obj_dir = os.path.join(root, "_cuda_build_objs")
    os.makedirs(obj_dir, exist_ok=True)

    common_includes = [
        f"-I{include_dir}",
        f"-I{cuda_include}",
    ]

    # ── Step 1: compile each .cu → .o with nvcc ─────────────────────
    obj_files = []
    print("Compiling CUDA kernels …")
    for cu in cu_files:
        basename = os.path.splitext(os.path.basename(cu))[0]
        obj = os.path.join(obj_dir, f"{basename}.o")
        obj_files.append(obj)

        cmd = [
            "nvcc",
            "-O3",
            "-arch=sm_89",
            "-std=c++17",
            "--compiler-options", "-fPIC",
            "-c", cu,
            "-o", obj,
        ] + common_includes

        print(f"  {os.path.basename(cu)} -> {os.path.basename(obj)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"\n nvcc failed on {os.path.basename(cu)}:\n{result.stderr}")
            sys.exit(1)

    # ── Step 2: compile cuda_bindings.cpp → .o with nvcc ─────────────
    bindings_obj = os.path.join(obj_dir, "cuda_bindings.o")
    obj_files.append(bindings_obj)

    cmd_bindings = [
        "nvcc",
        "-O3",
        "-std=c++17",
        "--compiler-options", "-fPIC",
        "-c", bindings_cpp,
        "-o", bindings_obj,
    ] + common_includes + [
        f"-I{python_include}",
        f"-I{numpy_include}",
        f"-I{pybind11_include}",
    ]

    print(f"  cuda_bindings.cpp → cuda_bindings.o")
    result = subprocess.run(cmd_bindings, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n✗ nvcc failed on cuda_bindings.cpp:\n{result.stderr}")
        sys.exit(1)

    # ── Step 3: link all .o → shared library ─────────────────────────
    print("Linking …")
    cmd_link = [
        "nvcc",
        "-shared",
        "--compiler-options", "-fPIC",
    ] + obj_files + [
        f"-L{cuda_lib}",
        "-lcudart",
        "-o", output,
    ]

    result = subprocess.run(cmd_link, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n✗ Linking failed:\n{result.stderr}")
        sys.exit(1)

    # ── Cleanup object files ─────────────────────────────────────────
    shutil.rmtree(obj_dir, ignore_errors=True)

    print(f"\n✓ Build successful: {output}")
    print(f"  Size: {os.path.getsize(output) / 1024:.0f} KB")

    # ── Quick import test ────────────────────────────────────────────
    sys.path.insert(0, root)
    try:
        import importlib
        mod = importlib.import_module("seera_cuda")
        print(f"✓ Import test passed: {mod.__doc__}")
    except Exception as e:
        print(f"✗ Import failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    build()
