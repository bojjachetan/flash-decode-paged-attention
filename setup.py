from pathlib import Path

from setuptools import find_packages, setup


def build_extensions():
    try:
        import torch
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME
    except Exception:
        return [], {}

    if CUDA_HOME is None:
        return [], {}

    ext = CUDAExtension(
        name="flash_decode._C",
        sources=[
            "csrc/bindings.cpp",
            "csrc/paged_attention_kernel.cu",
        ],
        extra_compile_args={
            "cxx": ["-O3"],
            "nvcc": [
                "-O3",
                "--use_fast_math",
                "-lineinfo",
                "-U__CUDA_NO_HALF_OPERATORS__",
                "-U__CUDA_NO_HALF_CONVERSIONS__",
                "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
            ],
        },
    )
    return [ext], {"build_ext": BuildExtension}


ext_modules, cmdclass = build_extensions()

setup(
    name="flash-decode-paged-attention",
    version="0.1.0",
    description="CUDA paged KV-cache decode attention kernel for LLM inference.",
    long_description=Path("README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=["torch>=2.0"],
    ext_modules=ext_modules,
    cmdclass=cmdclass,
)

