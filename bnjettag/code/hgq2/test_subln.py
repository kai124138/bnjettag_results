#!/usr/bin/env python
"""Verification of the SubLN hls4ml extension (keras-v3 -> Vitis, io_parallel C-sim).

Per shape class, builds [Input -> PSubLN] (BitExact forced on), converts with the
Vitis backend, patches the written project for macOS, compiles the C-sim library,
predicts 2048 realistic samples and reports max-abs-err + Pearson corr of the
SubLN output against the keras float reference. Data mimics the real model:
(10,16) raw jet features with per-token var ~1e4..1e6 (incl. zero-padded tokens),
(10,256)/(10,1024)/2-D (256,) hidden states with var ~0.1..100, plus the
flatten_axes=2 variant on (10,4,64).

Integration: PSubLN feeding hgq QDense (frozen binary {-1,+1} kernel) and hgq
QEinsumDense (the real per-token use) must convert, compile and stay finite.

Acceptance gate: corr >= 0.9999 per shape and no NaN/inf anywhere.

Run:  SUBLN_BUILD_DIR=<scratch> .venv-hgq2/bin/python test_subln.py
"""

import os
import platform
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')

import keras  # noqa: E402

from bnhgq2.compat import apply_hls4ml_compat, patch_project_for_macos  # noqa: E402
from bnhgq2.subln import PSubLN, register_subln, subln_output_kif  # noqa: E402

apply_hls4ml_compat()
register_subln()

import hls4ml  # noqa: E402

BUILD_DIR = Path(os.environ.get('SUBLN_BUILD_DIR', tempfile.mkdtemp(prefix='subln_'))).resolve()
BUILD_DIR.mkdir(parents=True, exist_ok=True)

N_SAMPLES = 2048
CORR_TARGET = 0.9999
RNG = np.random.default_rng(20260704)


def make_data(shape, kind, flatten_axes=1):
    """Realistic per-token data. kind='raw': var ~1e4..1e6 (+ zero-padded tokens);
    kind='hidden': var ~0.1..100."""
    tok_axes = len(shape) - flatten_axes  # leading (token) axes kept independent
    scale_shape = (N_SAMPLES,) + shape[:tok_axes] + (1,) * flatten_axes
    if kind == 'raw':
        sigma = RNG.uniform(100.0, 1000.0, size=scale_shape)  # var 1e4..1e6
        offset = RNG.uniform(-500.0, 500.0, size=scale_shape)
    else:
        sigma = RNG.uniform(np.sqrt(0.1), 10.0, size=scale_shape)  # var 0.1..100
        offset = RNG.uniform(-3.0, 3.0, size=scale_shape)
    x = offset + sigma * RNG.standard_normal((N_SAMPLES,) + shape)
    if kind == 'raw':
        x[: N_SAMPLES // 2, -3:, :] = 0.0  # zero-padded constituents (half the jets)
    return x.astype(np.float32)


def convert_compile(model, name, default_precision='fixed<18,8>'):
    out_dir = BUILD_DIR / name
    hls_config = {'Model': {'Precision': default_precision, 'ReuseFactor': 1}}
    hls_model = hls4ml.converters.convert_from_keras_model(
        model,
        output_dir=str(out_dir),
        project_name=name,
        backend='Vitis',
        io_type='io_parallel',
        hls_config=hls_config,
        bit_exact=True,  # NB: convert_from_keras_model overwrites hls_config['Model']['BitExact'] with this kwarg
    )
    hls_model.write()
    if platform.system() == 'Darwin':
        patched = patch_project_for_macos(out_dir)
        print(f'    [macOS patch] {patched or "already patched"}')
    hls_model._compile()  # compile() would re-write and undo the patch
    return hls_model


def metrics(y_ref, y_hls):
    y_ref = np.asarray(y_ref, dtype=np.float64).ravel()
    y_hls = np.asarray(y_hls, dtype=np.float64).ravel()
    finite = bool(np.isfinite(y_hls).all())
    max_err = float(np.max(np.abs(y_ref - y_hls)))
    corr = float(np.corrcoef(y_ref, y_hls)[0, 1])
    return corr, max_err, finite


def run_shape_case(label, shape, kind, flatten_axes=1):
    name = 'subln_' + label
    print(f'\n=== {label}: shape {shape}, {kind}, flatten_axes={flatten_axes} ===')
    t0 = time.time()
    x = make_data(shape, kind, flatten_axes)
    inp = keras.Input(shape=shape)
    out = PSubLN(flatten_axes=flatten_axes)(inp)
    model = keras.Model(inp, out)
    y_ref = model.predict(x, batch_size=512, verbose=0)

    hls_model = convert_compile(model, name)
    y_hls = np.asarray(hls_model.predict(x)).reshape(y_ref.shape)

    corr, max_err, finite = metrics(y_ref, y_hls)
    dim = int(np.prod(shape[len(shape) - flatten_axes:]))
    k, i, f = subln_output_kif(dim)
    print(f'    out precision fixed<{k + i + f},{k + i}> | corr={corr:.9f} max_err={max_err:.3e} '
          f'finite={finite} [{time.time() - t0:.1f}s]')
    return {'label': label, 'shape': shape, 'kind': kind, 'corr': corr, 'max_err': max_err,
            'finite': finite, 'pass': finite and corr >= CORR_TARGET}


def frozen_kif(k0, i0, f0, place='datalane'):
    from hgq.quantizer.config import QuantizerConfig

    return QuantizerConfig(
        'kif', place, k0=k0, i0=i0, f0=f0, round_mode='RND', overflow_mode='SAT', trainable=False
    )


def assign_binary_kernel(layer):
    w = layer._kernel
    w.assign(RNG.choice([-1.0, 1.0], size=w.shape).astype(np.float32))


def run_integration_qdense():
    """2-D pooled vector: Input(256) -> Quantizer -> PSubLN -> QDense(4, binary frozen)."""
    from hgq.layers import QDense
    from hgq.quantizer import Quantizer

    label = 'int_qdense'
    print(f'\n=== integration: (256,) -> Quantizer -> PSubLN -> QDense(4, binary frozen) ===')
    t0 = time.time()
    shape = (256,)
    sigma = RNG.uniform(np.sqrt(0.1), 10.0, size=(N_SAMPLES, 1))
    x = (RNG.uniform(-3, 3, size=(N_SAMPLES, 1)) + sigma * RNG.standard_normal((N_SAMPLES,) + shape)).astype(np.float32)

    inp = keras.Input(shape=shape)
    q = Quantizer(frozen_kif(1, 6, 10))(inp)
    ln = PSubLN()(q)
    out = QDense(
        4,
        kq_conf=frozen_kif(1, 1, 0, 'weight'),
        iq_conf=frozen_kif(1, 4, 10),
        enable_ebops=False,
    )(ln)
    model = keras.Model(inp, out)
    assign_binary_kernel(model.layers[-1])
    y_ref = model.predict(x, batch_size=512, verbose=0)

    hls_model = convert_compile(model, label, default_precision='fixed<26,12>')
    y_hls = np.asarray(hls_model.predict(x)).reshape(y_ref.shape)
    corr, max_err, finite = metrics(y_ref, y_hls)
    print(f'    corr={corr:.9f} max_err={max_err:.3e} finite={finite} [{time.time() - t0:.1f}s]')
    return {'label': label, 'shape': shape, 'kind': 'hidden+QDense', 'corr': corr,
            'max_err': max_err, 'finite': finite, 'pass': finite and corr >= CORR_TARGET}


def run_integration_qeinsumdense():
    """Real use: Input(10,16) -> Quantizer -> PSubLN -> QEinsumDense (binary frozen)."""
    from hgq.layers import QEinsumDense
    from hgq.quantizer import Quantizer

    label = 'int_qeinsum'
    print(f'\n=== integration: (10,16) raw -> Quantizer -> PSubLN -> QEinsumDense abc,cd->abd (binary frozen) ===')
    t0 = time.time()
    shape = (10, 16)
    x = make_data(shape, 'raw')
    x = np.clip(x, -4000.0, 4000.0)  # keep inside the frozen input quantizer range

    inp = keras.Input(shape=shape)
    q = Quantizer(frozen_kif(1, 13, 10))(inp)
    ln = PSubLN()(q)
    ed = QEinsumDense(
        'abc,cd->abd',
        output_shape=(10, 8),
        kq_conf=frozen_kif(1, 1, 0, 'weight'),
        iq_conf=frozen_kif(1, 4, 10),
        enable_ebops=False,
    )
    out = ed(ln)
    model = keras.Model(inp, out)
    assign_binary_kernel(ed)
    # exercise the keras-3.15 compat shim the hgq stack depends on
    fos = ed.full_output_shape
    assert tuple(fos[1:]) == (10, 8), f'full_output_shape compat shim broken: {fos}'
    y_ref = model.predict(x, batch_size=512, verbose=0)

    hls_model = convert_compile(model, label, default_precision='fixed<26,12>')
    y_hls = np.asarray(hls_model.predict(x)).reshape(y_ref.shape)
    corr, max_err, finite = metrics(y_ref, y_hls)
    print(f'    corr={corr:.9f} max_err={max_err:.3e} finite={finite} [{time.time() - t0:.1f}s]')
    return {'label': label, 'shape': shape, 'kind': 'raw+QEinsumDense', 'corr': corr,
            'max_err': max_err, 'finite': finite, 'pass': finite and corr >= CORR_TARGET}


def run_sanity_checks():
    """Cheap invariants: serialization, idempotent registration, built-in path untouched."""
    # PSubLN keras save/load round trip
    inp = keras.Input(shape=(10, 16))
    m = keras.Model(inp, PSubLN(epsilon=1e-6, flatten_axes=1)(inp))
    x = make_data((10, 16), 'hidden')[:64]
    path = BUILD_DIR / 'psubln_roundtrip.keras'
    m.save(path)
    m2 = keras.models.load_model(path)
    np.testing.assert_allclose(m.predict(x, verbose=0), m2.predict(x, verbose=0), rtol=1e-6, atol=1e-6)

    # register_subln is idempotent
    register_subln()
    register_subln()

    # the built-in LayerNormalization IR/QKeras path is untouched
    from hls4ml.model.layers import LayerNormalization, layer_map

    assert layer_map['LayerNormalization'] is LayerNormalization, 'built-in LayerNormalization was clobbered'
    from hls4ml.model.optimizer import get_backend_passes

    for b in ('vivado', 'vitis'):
        passes = set(get_backend_passes(b.capitalize()))
        assert f'{b}:subln_config_template' in passes and f'{b}:subln_function_template' in passes
    assert 'vivado:layernormalization_config_template' in set(get_backend_passes('Vivado'))
    print('sanity checks: PSubLN save/load OK, register_subln idempotent, built-in LayerNormalization intact')


def main():
    print(f'build dir: {BUILD_DIR}')
    print(f'keras {keras.__version__} | hls4ml {hls4ml.__version__} | {platform.platform()}')

    run_sanity_checks()

    results = []
    results.append(run_shape_case('raw_10x16', (10, 16), 'raw'))
    results.append(run_shape_case('hid_10x256', (10, 256), 'hidden'))
    results.append(run_shape_case('hid_10x1024', (10, 1024), 'hidden'))
    results.append(run_shape_case('vec_256', (256,), 'hidden'))
    results.append(run_shape_case('fa2_10x4x64', (10, 4, 64), 'hidden', flatten_axes=2))
    results.append(run_integration_qdense())
    results.append(run_integration_qeinsumdense())

    print('\n' + '=' * 78)
    print(f'{"case":<14} {"shape":<12} {"kind":<18} {"corr":>12} {"max_err":>11} {"finite":>7} {"pass":>5}')
    print('-' * 78)
    for r in results:
        print(f'{r["label"]:<14} {str(r["shape"]):<12} {r["kind"]:<18} {r["corr"]:>12.9f} '
              f'{r["max_err"]:>11.3e} {str(r["finite"]):>7} {"OK" if r["pass"] else "FAIL":>5}')
    print('=' * 78)
    print(f'target: corr >= {CORR_TARGET}, no NaN/inf')

    if not all(r['pass'] for r in results):
        sys.exit(1)
    print('ALL PASS')


if __name__ == '__main__':
    main()
