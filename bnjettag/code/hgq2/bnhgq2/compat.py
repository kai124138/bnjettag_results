"""Compatibility shims: keras 3.15 / hgq2 0.1.9 / hls4ml 1.3.0 / macOS C-sim.

Three known impedance mismatches in this stack:

1. keras 3.15 removed ``EinsumDense.full_output_shape`` (it used to be stored by
   ``build()``); hgq2 0.1.9 still reads it (``hgq/layers/core/einsum_dense.py``,
   ``hgq/layers/attn/mha.py``). We add it back as a property that recomputes the
   shape from the einsum equation and the recorded build input shape.

2. hls4ml 1.3.0 registered its QMultiHeadAttention keras-v3 handler under the key
   ``hgq.layers.multi_head_attention.QMultiHeadAttention``, but in hgq 0.1.9 the
   class actually lives in ``hgq.layers.attn.mha`` (dispatch key is
   ``f'{cls.__module__}.{cls.__name__}'``). We alias the real key to the
   registered handler. Same treatment for QLinformerAttention if present.

3. The Xilinx ap_types headers forward-declare ``std::complex`` instead of
   including <complex> (an AP_AUTOCC workaround). Apple clang/libc++ rejects the
   later explicit specializations. ``patch_project_for_macos`` rewrites the two
   offending headers in a *written project* so ``compile()``/C-sim works locally.
   This only touches the generated project directory — never the installed
   hls4ml templates — and is a no-op for csynth on mulder (Linux).

Call ``apply_hls4ml_compat()`` once, before building/converting models, and
``patch_project_for_macos(project_dir)`` after ``model.write()`` / before
``model._compile()`` (``compile()`` re-writes the project, so patch in between).
"""

from __future__ import annotations

import re
from pathlib import Path

_STD_COMPLEX_FWD = re.compile(r'namespace std \{\s*template\s*<\s*typename _Tp\s*>\s*class complex;\s*\}')


def apply_keras_compat() -> None:
    """Keras-level shims only (no hls4ml import) — EBOPs/tracing needs these too."""
    _patch_einsum_dense_full_output_shape()


def apply_hls4ml_compat() -> None:
    """Apply all in-process shims. Idempotent."""
    _patch_einsum_dense_full_output_shape()
    _alias_hgq_attention_handlers()


def _patch_einsum_dense_full_output_shape() -> None:
    from keras.src.layers.core.einsum_dense import EinsumDense, _analyze_einsum_string

    if hasattr(EinsumDense, 'full_output_shape'):
        return

    def full_output_shape(self):
        if not self.built:
            raise AttributeError(f'EinsumDense layer {self.name}: full_output_shape requires the layer to be built')
        input_shape = self._build_shapes_dict['input_shape']
        shape_data = _analyze_einsum_string(self.equation, self.bias_axes, input_shape, self.partial_output_shape)
        return tuple(shape_data[2])  # (kernel_shape, bias_shape, full_output_shape, ...)

    EinsumDense.full_output_shape = property(full_output_shape)


def _alias_hgq_attention_handlers() -> None:
    import hgq
    from hls4ml.converters.keras_v3._base import registry

    stale_by_class = {
        'QMultiHeadAttention': 'hgq.layers.multi_head_attention.QMultiHeadAttention',
        'QLinformerAttention': 'hgq.layers.linformer_attention.QLinformerAttention',
    }
    for cls_name, stale_key in stale_by_class.items():
        cls = getattr(hgq.layers, cls_name, None)
        if cls is None:
            continue
        real_key = f'{cls.__module__}.{cls.__name__}'
        if real_key not in registry and stale_key in registry:
            registry[real_key] = registry[stale_key]


def patch_project_for_macos(project_dir: str | Path) -> list[str]:
    """Fix ap_int_special.h / ap_fixed_special.h in a written hls4ml project.

    Replaces the forward declaration of std::complex with ``#include <complex>``
    so Apple clang (libc++) accepts the explicit specializations. Returns the
    list of files patched (empty if already patched or not present).
    """
    ap_dir = Path(project_dir) / 'firmware' / 'ap_types'
    patched = []
    for fname in ('ap_int_special.h', 'ap_fixed_special.h'):
        path = ap_dir / fname
        if not path.is_file():
            continue
        text = path.read_text()
        new_text, n_subs = _STD_COMPLEX_FWD.subn('#include <complex>', text)
        if n_subs:
            path.write_text(new_text)
            patched.append(fname)
    return patched
