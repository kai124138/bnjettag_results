"""SubLN — parameter-free per-token LayerNorm as an hls4ml custom layer.

    y = (x - mean(x, last axes)) / sqrt(var(x, last axes) + eps)

Biased variance, NO learnable gamma/beta (the BitNet "SubLN" placement). Used 51
times in the transformer, on (B,10,16) raw features (var up to ~1e6), (B,10,256)
and (B,10,1024) hidden states, and 2-D (B,256) pooled vectors.

Why not hls4ml's built-in LayerNormalization: its inverse-sqrt table is indexed
directly by the variance and only covers var in (eps, 1] — wrong for everything
this model normalizes. The kernel here (hls_templates/nnet_subln.h) range-reduces
the variance by an even power of two onto [1/4, 1) first, so one 4096-entry table
serves any variance magnitude exactly up to table resolution.

Pieces (see hls4ml docs/advanced/extension.rst for the pattern):
  * PSubLN               — the keras (v3) layer, pure keras.ops, serializable.
  * PSubLNHandler        — keras-v3 frontend handler (auto-registered by metaclass).
  * SubLN                — hls4ml IR layer (new class; the built-in
                           LayerNormalization IR/kernel is left untouched so the
                           QKeras path keeps working).
  * _produce_kif/_request_kif registrations — the BitExact pass has no rule for
    LayerNormalization-like layers and would raise NotImplementedError otherwise.
  * SubLNConfigTemplate / SubLNFunctionTemplate — Vivado/Vitis codegen. Internal
    C++ types are derived per layer from the *final* input precision at codegen
    time, so they are correct with or without the BitExact flow.
  * register_subln()     — idempotent registration of all of the above plus the
    C++ source (backend.register_source).

Usage:
    from bnhgq2.compat import apply_hls4ml_compat
    from bnhgq2.subln import PSubLN, register_subln
    apply_hls4ml_compat()
    register_subln()
    ... build keras model with PSubLN ... convert_from_keras_model(backend='Vitis')
"""

from __future__ import annotations

import math
from pathlib import Path

import keras
import numpy as np
from keras import ops

HLS_TEMPLATE_DIR = Path(__file__).resolve().parent.parent / 'hls_templates'

# Fixed output precision declared to the BitExact pass (tuned in test_subln.py).
# Total width 18; integer bits follow the mathematical bound |y| <= sqrt(dim-1).
SUBLN_OUT_WIDTH = 18
# What SubLN asks of its input: fixed<31,15> equivalent (k=1, i=14, f=16).
# Covers raw jet features to |x| < 16384 at 2^-16 resolution; any narrower
# upstream quantizer wins the min() in the BitExact flow anyway.
SUBLN_REQ_I = 14
SUBLN_REQ_F = 16
# Internal fractional widths of the C++ kernel (see SubLNConfigTemplate.format)
SUBLN_VAR_F = 26  # fractional bits of the variance; sets the smallest resolvable var+eps
SUBLN_PROD_F = 24  # fractional bits of diff * invsqrt before the exponent shift
SUBLN_TABLE_W = 18  # 1/sqrt table entries: ap_ufixed<18,1>, values in (1,2)


# --------------------------------------------------------------------------- #
# (a) keras layer                                                             #
# --------------------------------------------------------------------------- #
@keras.saving.register_keras_serializable(package='bnhgq2')
class PSubLN(keras.layers.Layer):
    """Parameter-free sub-layer norm over the last `flatten_axes` axes.

    y = (x - mean) / sqrt(var + epsilon), biased variance, no gamma/beta.
    flatten_axes=1 normalizes the last axis (default); flatten_axes=2 normalizes
    over the last two axes jointly (e.g. (B,T,H,E) -> per-(T) over H*E).
    """

    def __init__(self, epsilon: float = 1e-6, flatten_axes: int = 1, **kwargs):
        super().__init__(**kwargs)
        if flatten_axes < 1:
            raise ValueError(f'flatten_axes must be >= 1, got {flatten_axes}')
        self.epsilon = float(epsilon)
        self.flatten_axes = int(flatten_axes)

    def call(self, x):
        axes = tuple(range(-self.flatten_axes, 0))
        mean = ops.mean(x, axis=axes, keepdims=True)
        var = ops.mean(ops.square(x - mean), axis=axes, keepdims=True)  # biased
        return (x - mean) / ops.sqrt(var + self.epsilon)

    def build(self, input_shape):
        if len(input_shape) - 1 < self.flatten_axes:  # batch + at least flatten_axes
            raise ValueError(
                f'{self.name}: input rank {len(input_shape)} too small for flatten_axes={self.flatten_axes}'
            )
        super().build(input_shape)

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super().get_config()
        config.update({'epsilon': self.epsilon, 'flatten_axes': self.flatten_axes})
        return config


# --------------------------------------------------------------------------- #
# (c) hls4ml IR layer                                                         #
# --------------------------------------------------------------------------- #
from hls4ml.model.attributes import Attribute  # noqa: E402
from hls4ml.model.layers import Layer as _HLS4MLLayer  # noqa: E402


class SubLN(_HLS4MLLayer):
    """hls4ml IR node for PSubLN. No weights; dim = n_in / seq_len."""

    _expected_attributes = [
        Attribute('n_in'),
        Attribute('seq_len'),
        Attribute('epsilon', value_type=float, default=1e-6),
        Attribute('table_size', default=4096, configurable=True),
    ]

    def initialize(self):
        inp = self.get_input_variable()
        self.add_output_variable(list(inp.shape))


# --------------------------------------------------------------------------- #
# (b) keras-v3 frontend handler (auto-registered via the `handles` metaclass)  #
# --------------------------------------------------------------------------- #
from hls4ml.converters.keras_v3._base import KerasV3LayerHandler  # noqa: E402


class PSubLNHandler(KerasV3LayerHandler):
    """keras-v3 handler: PSubLN -> SubLN IR config."""

    # exact key is '<module>.<class>'; bare class name is the dispatcher fallback
    handles = (f'{__name__}.PSubLN', 'PSubLN')

    def handle(self, layer, in_tensors, out_tensors):
        assert len(in_tensors) == 1 and len(out_tensors) == 1, f'{layer.name}: PSubLN must be single-input/single-output'
        shape = tuple(in_tensors[0].shape[1:])
        assert all(d is not None for d in shape), f'{layer.name}: PSubLN needs fully-defined input shape, got {shape}'
        fa = int(layer.flatten_axes)
        assert 1 <= fa <= len(shape), f'{layer.name}: flatten_axes={fa} invalid for input shape {shape}'
        n_in = int(np.prod(shape))
        seq_len = int(np.prod(shape[:-fa])) if len(shape) > fa else 1
        return {
            'class_name': 'SubLN',
            'n_in': n_in,
            'seq_len': seq_len,
            'epsilon': float(layer.epsilon),
        }


# --------------------------------------------------------------------------- #
# (d) BitExact pass hooks: produce/request kif                                #
# --------------------------------------------------------------------------- #
from hls4ml.model.optimizer.passes.bit_exact import (  # noqa: E402
    _produce_kif,
    _request_kif,
    get_input_shapes,
    get_output_shape,
)


def subln_output_kif(dim: int) -> tuple[int, int, int]:
    """(k, i, f) declared for the SubLN output. |y| <= sqrt(dim-1) exactly."""
    i = max(1, math.ceil(0.5 * math.log2(max(dim - 1, 2))))
    f = SUBLN_OUT_WIDTH - 1 - i
    return 1, i, f


@_produce_kif.register(SubLN)
def _(layer: SubLN):
    shape = get_output_shape(layer)
    dim = int(layer.attributes['n_in']) // int(layer.attributes['seq_len'])
    k, i, f = subln_output_kif(dim)
    return (
        np.full(shape, k, dtype=np.int16),
        np.full(shape, i, dtype=np.int16),
        np.full(shape, f, dtype=np.int16),
    )


@_request_kif.register(SubLN)
def _(layer: SubLN):
    # Finite request: bounds the precision of a layer/input feeding SubLN and
    # keeps FixInputPrecision from tripping over an unbounded input chain.
    shape = get_input_shapes(layer)[0]
    return (
        (
            np.ones(shape, dtype=np.int16),
            np.full(shape, SUBLN_REQ_I, dtype=np.int16),
            np.full(shape, SUBLN_REQ_F, dtype=np.int16),
        ),
    )


# --------------------------------------------------------------------------- #
# (e) Vivado/Vitis templates                                                  #
# --------------------------------------------------------------------------- #
from hls4ml.backends.template import FunctionCallTemplate, LayerConfigTemplate  # noqa: E402

subln_config_template = """struct config{index} : nnet::subln_config {{
    static const unsigned n_in = {n_in};
    static const unsigned seq_len = {seq_len};
    static const unsigned table_size = {table_size};
    static const unsigned table_size_log2 = {table_size_log2};
    static constexpr double epsilon = {epsilon_cpp};
    typedef {accum_t_cpp} accum_t;
    typedef {diff_t_cpp} diff_t;
    typedef {var_t_cpp} var_t;
    typedef {table_t_cpp} table_t;
    typedef {prod_t_cpp} prod_t;
    static const unsigned io_type = nnet::{iotype};
    static const unsigned reuse_factor = {reuse};
}};\n"""

subln_function_template = 'nnet::subln<{input_t}, {output_t}, {config}>({input}, {output});'

subln_include_list = ['nnet_utils/nnet_subln.h']


class SubLNConfigTemplate(LayerConfigTemplate):
    def __init__(self):
        super().__init__(SubLN)
        self.template = subln_config_template

    def format(self, node):
        params = self._default_config_params(node)

        if params['iotype'] != 'io_parallel':
            raise RuntimeError(f'SubLN ({node.name}): only io_parallel is supported, got {params["iotype"]}')

        n_in = int(node.get_attr('n_in'))
        seq_len = int(node.get_attr('seq_len'))
        dim = n_in // seq_len
        L = max(1, math.ceil(math.log2(dim)))  # accumulation growth bits

        table_size = int(node.get_attr('table_size'))
        table_size_log2 = int(round(math.log2(table_size)))
        if 2**table_size_log2 != table_size:
            raise RuntimeError(f'SubLN ({node.name}): table_size must be a power of 2, got {table_size}')

        # Derive internal widths from the FINAL input precision (post-optimizers).
        prec = node.get_input_variable().type.precision
        W_in = int(prec.width)
        I_in = int(prec.integer)  # includes sign bit when signed
        F_in = W_in - I_in
        signed = bool(getattr(prec, 'signed', True))
        Ii = I_in + (0 if signed else 1)  # equivalent signed integer bits

        # sum of dim inputs, and exact mean for power-of-two dim
        I_acc, F_acc = Ii + L + 1, F_in + L
        # centered value x - mean: |d| <= spread <= 2^Ii
        I_d, F_d = Ii + 1, F_in + L
        # variance (unsigned): sum(d^2)/dim <= 2^(2*I_d - 2 + L), +1 bit guard
        I_var, F_var = 2 * I_d - 1 + L, max(SUBLN_VAR_F, table_size_log2 + 2)
        # d * table (table < 2), plus headroom for the left-shift path (|y| < 2^7)
        I_p, F_p = max(I_d + 2, 8), SUBLN_PROD_F

        params['table_size_log2'] = table_size_log2
        params['epsilon_cpp'] = f'{float(node.get_attr("epsilon")):.12e}'
        params['accum_t_cpp'] = f'ap_fixed<{I_acc + F_acc},{I_acc}>'
        params['diff_t_cpp'] = f'ap_fixed<{I_d + F_d},{I_d}>'
        params['var_t_cpp'] = f'ap_ufixed<{I_var + F_var},{I_var}>'
        params['table_t_cpp'] = f'ap_ufixed<{SUBLN_TABLE_W},1>'
        params['prod_t_cpp'] = f'ap_fixed<{I_p + F_p},{I_p}>'

        return self.template.format(**params)


class SubLNFunctionTemplate(FunctionCallTemplate):
    def __init__(self):
        super().__init__(SubLN, include_header=subln_include_list)
        self.template = subln_function_template

    def format(self, node):
        params = self._default_function_params(node)
        return self.template.format(**params)


# --------------------------------------------------------------------------- #
# (f) registration                                                            #
# --------------------------------------------------------------------------- #
def register_subln(backends: tuple[str, ...] = ('Vivado', 'Vitis')) -> None:
    """Register the SubLN IR layer, codegen templates and C++ source. Idempotent.

    The keras-v3 handler and the BitExact produce/request hooks are registered
    at import time of this module; this adds everything hls4ml looks up by name.
    """
    from hls4ml.backends import get_backend
    from hls4ml.model.layers import layer_map, register_layer
    from hls4ml.model.optimizer import get_backend_passes

    if layer_map.get('SubLN') is not SubLN:
        register_layer('SubLN', SubLN)

    source = HLS_TEMPLATE_DIR / 'nnet_subln.h'
    if not source.is_file():
        raise FileNotFoundError(f'SubLN HLS source not found: {source}')

    for backend_name in backends:
        backend = get_backend(backend_name)
        existing = set(get_backend_passes(backend_name))
        for template_cls in (SubLNConfigTemplate, SubLNFunctionTemplate):
            pass_name = f'{backend_name.lower()}:{template_cls().get_name()}'
            if pass_name not in existing:
                backend.register_template(template_cls)
        # keyed by destination path -> re-registration just overwrites
        backend.register_source(str(source))
