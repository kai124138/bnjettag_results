#ifndef NNET_SUBLN_H_
#define NNET_SUBLN_H_

// Parameter-free per-token LayerNorm ("SubLN") for hls4ml (Vivado/Vitis, io_parallel).
//
//   y = (x - mean(x)) / sqrt(var(x) + eps)        (biased variance, no gamma/beta)
//
// The shipped nnet_layernorm.h indexes its 1/sqrt table directly with the variance,
// so it is only correct for var in (eps, 1]. Raw jet features have var up to ~1e6.
// This kernel RANGE-REDUCES the variance first:
//
//   var = m * 2^e2  with even e2 and m in [1/4, 1)
//   1/sqrt(var) = (1/sqrt(m)) * 2^(-e2/2),   1/sqrt(m) in (1, 2]
//
// The table only ever covers m in [1/4, 1), the power-of-two part is an exact shift.
// All internal types are generated per-layer from the input precision (see
// bnhgq2/subln.py::SubLNConfigTemplate); the defaults below are placeholders.

#include "nnet_common.h"
#include <math.h>

namespace nnet {

struct subln_config {
    // Internal data types (overridden by the generated per-layer config)
    typedef ap_fixed<24, 8> accum_t;    // sum / mean
    typedef ap_fixed<32, 9> diff_t;     // centered values x - mean
    typedef ap_ufixed<48, 22> var_t;    // variance (unsigned); fractional part must be >= table_size_log2
    typedef ap_ufixed<18, 1> table_t;   // 1/sqrt(m), m in [1/4,1) -> values in (1, 2)
    typedef ap_fixed<40, 10> prod_t;    // diff * table value, before the 2^(-e2/2) shift

    // Layer sizes
    static const unsigned n_in = 16;    // total elements
    static const unsigned seq_len = 1;  // number of tokens; dim = n_in / seq_len

    // Inverse-sqrt table
    static const unsigned table_size = 4096;
    static const unsigned table_size_log2 = 12;

    static constexpr double epsilon = 1e-6;

    // Resource reuse info
    static const unsigned io_type = io_parallel;
    static const unsigned reuse_factor = 1;
};

// Table over m in [0,1): entry i covers [i/N, (i+1)/N), built at the bin midpoint.
// Only i >= N/4 is ever addressed (m in [1/4,1)); lower entries are zeroed.
template <typename CONFIG_T, int N_TABLE> void init_subln_invsqrt_table(typename CONFIG_T::table_t table_out[N_TABLE]) {
    for (int i = 0; i < N_TABLE; i++) {
        if (4 * i < N_TABLE) {
            table_out[i] = (typename CONFIG_T::table_t)0;
        } else {
            double m = (i + 0.5) / (double)N_TABLE;
            table_out[i] = (typename CONFIG_T::table_t)(1.0 / sqrt(m)); // < 2, fits ap_ufixed<w,1>
        }
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void subln_1d(data_T data[CONFIG_T::n_in / CONFIG_T::seq_len], res_T res[CONFIG_T::n_in / CONFIG_T::seq_len],
              typename CONFIG_T::table_t invsqrt_table[CONFIG_T::table_size]) {
    #pragma HLS PIPELINE II=CONFIG_T::reuse_factor
    #pragma HLS ARRAY_PARTITION variable=data complete
    #pragma HLS ARRAY_PARTITION variable=res complete

    typedef typename CONFIG_T::accum_t accum_t;
    typedef typename CONFIG_T::diff_t diff_t;
    typedef typename CONFIG_T::var_t var_t;
    typedef typename CONFIG_T::table_t table_t;
    typedef typename CONFIG_T::prod_t prod_t;

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;
    static const int VW = var_t::width;
    static const int VI = var_t::iwidth;
    static const int VF = VW - VI; // fractional bits of var_t

    // ---- 1) mean -----------------------------------------------------------
    accum_t sum = 0;
SUBLN_SUM:
    for (unsigned i = 0; i < dim; i++) {
        #pragma HLS UNROLL
        sum += (accum_t)data[i];
    }
    const accum_t k_inv = 1.0 / (double)dim; // exact for power-of-two dim
    accum_t mean = sum * k_inv;

    // ---- 2) centered values + biased variance ------------------------------
    // The square d*d is the variance's only multiply. On defaults Vitis maps each
    // (diff_t x diff_t) to a DSP48 (measured whole_model_rf1: mul_36s_36s -> 4 DSP
    // each, the dominant SubLN DSP cost). sq_t is the EXACT full-width product type
    // ap_fixed<2W,2I>, so binding it to LUT fabric is numerically identical to the
    // original (var_t)(d[i]*d[i]) -- it only moves the operator off the DSP column.
    typedef ap_fixed<2 * diff_t::width, 2 * diff_t::iwidth> sq_t;
    diff_t d[dim];
    #pragma HLS ARRAY_PARTITION variable=d complete
    var_t s2 = 0;
SUBLN_VAR:
    for (unsigned i = 0; i < dim; i++) {
        #pragma HLS UNROLL
        d[i] = (diff_t)((accum_t)data[i] - mean);
        sq_t sq = d[i] * d[i];
        #pragma HLS BIND_OP variable=sq op=mul impl=fabric
        s2 += (var_t)sq;
    }
    const var_t vk_inv = 1.0 / (double)dim;
    var_t var = s2 * vk_inv;
    var += (var_t)CONFIG_T::epsilon;

    // ---- 3) range reduction: var = m * 2^e2, e2 even, m in [1/4, 1) --------
    ap_uint<VW> vbits = var.range(VW - 1, 0);
    int msb = -1; // index of highest set bit
SUBLN_MSB:
    for (int b = 0; b < VW; b++) {
        #pragma HLS UNROLL
        if (vbits[b])
            msb = b;
    }
    // bit b has weight 2^(b - VF) => floor(log2(var)) = msb - VF
    int e2 = msb - VF + 1; // var * 2^-e2 in [1/2, 1)
    if (e2 & 1)
        e2 += 1;           // make even    => var * 2^-e2 in [1/4, 1)
    var_t vnorm = var >> e2; // ap_fixed handles negative shift amounts (-> left shift)

    // ---- 4) table lookup: 1/sqrt(m), m in [1/4, 1) --------------------------
    // index = top table_size_log2 fractional bits of vnorm (vnorm < 1 by construction)
    ap_uint<CONFIG_T::table_size_log2> idx = vnorm.range(VF - 1, VF - CONFIG_T::table_size_log2);
    table_t inv_norm = (msb < 0) ? (table_t)0 : invsqrt_table[idx];
    int h = (msb < 0) ? 0 : (e2 >> 1); // exact halving: e2 is even (also for negative e2)

    // ---- 5) y_i = d_i * (1/sqrt(m)) * 2^-h ----------------------------------
    // scale_t is the EXACT product type of diff_t * table_t (ap_fixed<Wd+Wt, Id+It>);
    // binding it to LUT fabric matches the original d[i]*inv_norm exactly while keeping
    // the scale multiply off the DSP column (measured: mul_36s_18ns -> 2 DSP each).
    typedef ap_fixed<diff_t::width + table_t::width, diff_t::iwidth + table_t::iwidth> scale_t;
SUBLN_RES:
    for (unsigned i = 0; i < dim; i++) {
        #pragma HLS UNROLL
        scale_t praw = d[i] * inv_norm;
        #pragma HLS BIND_OP variable=praw op=mul impl=fabric
        prod_t p = praw;
        prod_t ps = p >> h; // negative h -> left shift; |y| <= sqrt(dim-1), fits prod_t
        res[i] = ps;        // final cast applies res_T rounding / saturation
    }
}

template <class data_T, class res_T, typename CONFIG_T>
void subln(data_T data[CONFIG_T::n_in], res_T res[CONFIG_T::n_in]) {
    static_assert(CONFIG_T::n_in % CONFIG_T::seq_len == 0, "SubLN: n_in must be divisible by seq_len");
    static_assert((1u << CONFIG_T::table_size_log2) == CONFIG_T::table_size, "SubLN: table_size must be 2^table_size_log2");
    static_assert((int)CONFIG_T::var_t::width - (int)CONFIG_T::var_t::iwidth >= (int)CONFIG_T::table_size_log2,
                  "SubLN: var_t needs at least table_size_log2 fractional bits");

    static const unsigned dim = CONFIG_T::n_in / CONFIG_T::seq_len;

#ifdef __HLS_SYN__
    bool initialized = false;
    typename CONFIG_T::table_t invsqrt_table[CONFIG_T::table_size];
#else
    static bool initialized = false;
    static typename CONFIG_T::table_t invsqrt_table[CONFIG_T::table_size];
#endif
    if (!initialized) {
        init_subln_invsqrt_table<CONFIG_T, CONFIG_T::table_size>(invsqrt_table);
        initialized = true;
    }

    data_T in_val[dim];
    res_T out_val[dim];
    #pragma HLS ARRAY_PARTITION variable=in_val complete
    #pragma HLS ARRAY_PARTITION variable=out_val complete

SUBLN_SEQ:
    for (unsigned j = 0; j < CONFIG_T::seq_len; j++) {
        #pragma HLS PIPELINE
    SUBLN_LOAD:
        for (unsigned i = 0; i < dim; i++) {
            #pragma HLS UNROLL
            in_val[i] = data[j * dim + i];
        }
        subln_1d<data_T, res_T, CONFIG_T>(in_val, out_val, invsqrt_table);
    SUBLN_STORE:
        for (unsigned i = 0; i < dim; i++) {
            #pragma HLS UNROLL
            res[j * dim + i] = out_val[i];
        }
    }
}

} // namespace nnet

#endif
