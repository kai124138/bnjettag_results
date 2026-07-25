#ifndef DEFINES_H_
#define DEFINES_H_

#include "ap_fixed.h"
#include "ap_int.h"
#include "nnet_utils/nnet_types.h"
#include <array>
#include <cstddef>
#include <cstdio>
#include <tuple>
#include <tuple>


// hls-fpga-machine-learning insert numbers

// hls-fpga-machine-learning insert layer-precision
typedef ap_fixed<8,4,AP_RND_CONV,AP_SAT,0> input_1_t;
typedef ap_fixed<25,9> input_proj_accum_t;
typedef ap_fixed<16,9,AP_RND_CONV,AP_SAT,0> input_proj_t;
typedef ap_fixed<2,2> input_proj_weight_t;
typedef ap_fixed<19,3> input_proj_bias_t;
typedef ap_fixed<19,7> input_proj_affine_t;
typedef ap_ufixed<3,-2> input_proj_affine_scale_t;
typedef ap_ufixed<2,32> input_proj_affine_bias_t;
typedef ap_fixed<8,4> bit_block_0_attn_Wq_iq_t;
typedef ap_fixed<13,9> bit_block_0_attn_Wq_accum_t;
typedef ap_fixed<13,9,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_Wq_t;
typedef ap_fixed<24,12> model_default_t;
typedef ap_fixed<8,4> bit_block_0_attn_Wk_iq_t;
typedef ap_fixed<13,9> bit_block_0_attn_Wk_accum_t;
typedef ap_fixed<13,9,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_Wk_t;
typedef ap_fixed<29,21> bit_block_0_attn_scores_accum_t;
typedef ap_fixed<29,21> bit_block_0_attn_scores_t;
typedef ap_ufixed<12,1,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_softmax_exp_table_t;
typedef ap_ufixed<12,1,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_softmax_inv_table_t;
typedef ap_ufixed<12,4,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_softmax_inv_inp_t;
typedef ap_ufixed<10,13,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_softmax_inp_norm_t;
typedef ap_ufixed<27,8> bit_block_0_attn_softmax_accum_t;
typedef ap_ufixed<23,1,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_softmax_t;
typedef ap_fixed<18,8> bit_block_0_attn_softmax_table_t;
typedef ap_fixed<8,4> bit_block_0_attn_Wv_iq_t;
typedef ap_fixed<13,9> bit_block_0_attn_Wv_accum_t;
typedef ap_fixed<13,9,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_Wv_t;
typedef ap_fixed<40,14> bit_block_0_attn_ctx_accum_t;
typedef ap_fixed<15,8,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_ctx_t;
typedef ap_fixed<20,13> bit_block_0_attn_Wo_accum_t;
typedef ap_fixed<15,8,AP_RND_CONV,AP_SAT,0> bit_block_0_attn_Wo_t;
typedef ap_fixed<19,3> bit_block_0_attn_Wo_affine_t;
typedef ap_ufixed<2,-5> bit_block_0_attn_Wo_affine_scale_t;
typedef ap_fixed<15,-1> bit_block_0_attn_Wo_affine_bias_t;
typedef ap_fixed<24,8> bit_block_0_add_attn_t;
typedef ap_fixed<8,4> bit_block_0_ffn_fc1_iq_t;
typedef ap_fixed<26,10> bit_block_0_ffn_fc1_accum_t;
typedef ap_fixed<26,10> bit_block_0_ffn_fc1_t;
typedef ap_fixed<2,2> bit_block_0_ffn_fc1_weight_t;
typedef ap_fixed<18,2> bit_block_0_ffn_fc1_bias_t;
typedef ap_ufixed<15,8,AP_RND_CONV,AP_SAT,0> bit_block_0_ffn_act_t;
typedef ap_ufixed<2,32> bit_block_0_ffn_act_param_t;
typedef ap_fixed<18,8> bit_block_0_ffn_act_table_t;
typedef ap_fixed<22,15> bit_block_0_ffn_fc2_accum_t;
typedef ap_fixed<16,9,AP_RND_CONV,AP_SAT,0> bit_block_0_ffn_fc2_t;
typedef ap_fixed<2,2> bit_block_0_ffn_fc2_weight_t;
typedef ap_ufixed<2,32> bit_block_0_ffn_fc2_bias_t;
typedef ap_fixed<20,4> bit_block_0_ffn_fc2_affine_t;
typedef ap_ufixed<3,-5> bit_block_0_ffn_fc2_affine_scale_t;
typedef ap_fixed<15,-1> bit_block_0_ffn_fc2_affine_bias_t;
typedef ap_fixed<25,9> bit_block_0_add_ffn_t;
typedef ap_fixed<8,4> bit_block_1_attn_Wq_iq_t;
typedef ap_fixed<13,9> bit_block_1_attn_Wq_accum_t;
typedef ap_fixed<13,9,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_Wq_t;
typedef ap_fixed<8,4> bit_block_1_attn_Wk_iq_t;
typedef ap_fixed<13,9> bit_block_1_attn_Wk_accum_t;
typedef ap_fixed<13,9,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_Wk_t;
typedef ap_fixed<29,21> bit_block_1_attn_scores_accum_t;
typedef ap_fixed<29,21> bit_block_1_attn_scores_t;
typedef ap_ufixed<12,1,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_softmax_exp_table_t;
typedef ap_ufixed<12,1,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_softmax_inv_table_t;
typedef ap_ufixed<12,4,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_softmax_inv_inp_t;
typedef ap_ufixed<10,14,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_softmax_inp_norm_t;
typedef ap_ufixed<27,8> bit_block_1_attn_softmax_accum_t;
typedef ap_ufixed<23,1,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_softmax_t;
typedef ap_fixed<18,8> bit_block_1_attn_softmax_table_t;
typedef ap_fixed<8,4> bit_block_1_attn_Wv_iq_t;
typedef ap_fixed<13,9> bit_block_1_attn_Wv_accum_t;
typedef ap_fixed<13,9,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_Wv_t;
typedef ap_fixed<40,14> bit_block_1_attn_ctx_accum_t;
typedef ap_fixed<16,9,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_ctx_t;
typedef ap_fixed<21,14> bit_block_1_attn_Wo_accum_t;
typedef ap_fixed<17,10,AP_RND_CONV,AP_SAT,0> bit_block_1_attn_Wo_t;
typedef ap_fixed<21,5> bit_block_1_attn_Wo_affine_t;
typedef ap_ufixed<3,-5> bit_block_1_attn_Wo_affine_scale_t;
typedef ap_fixed<15,-1> bit_block_1_attn_Wo_affine_bias_t;
typedef ap_fixed<26,10> bit_block_1_add_attn_t;
typedef ap_fixed<8,5> bit_block_1_ffn_fc1_iq_t;
typedef ap_fixed<26,10> bit_block_1_ffn_fc1_accum_t;
typedef ap_fixed<26,10> bit_block_1_ffn_fc1_t;
typedef ap_fixed<2,2> bit_block_1_ffn_fc1_weight_t;
typedef ap_fixed<18,2> bit_block_1_ffn_fc1_bias_t;
typedef ap_ufixed<16,9,AP_RND_CONV,AP_SAT,0> bit_block_1_ffn_act_t;
typedef ap_ufixed<2,32> bit_block_1_ffn_act_param_t;
typedef ap_fixed<18,8> bit_block_1_ffn_act_table_t;
typedef ap_fixed<23,16> bit_block_1_ffn_fc2_accum_t;
typedef ap_fixed<18,11,AP_RND_CONV,AP_SAT,0> bit_block_1_ffn_fc2_t;
typedef ap_fixed<2,2> bit_block_1_ffn_fc2_weight_t;
typedef ap_ufixed<2,32> bit_block_1_ffn_fc2_bias_t;
typedef ap_fixed<22,6> bit_block_1_ffn_fc2_affine_t;
typedef ap_ufixed<3,-5> bit_block_1_ffn_fc2_affine_scale_t;
typedef ap_fixed<15,-1> bit_block_1_ffn_fc2_affine_bias_t;
typedef ap_fixed<27,11> bit_block_1_add_ffn_t;
typedef ap_fixed<63,15> gap_accum_t;
typedef ap_fixed<8,4,AP_RND_CONV,AP_SAT,0> gap_t;
typedef ap_fixed<26,10> head_fc1_accum_t;
typedef ap_fixed<26,10> head_fc1_t;
typedef ap_fixed<2,2> head_fc1_weight_t;
typedef ap_fixed<17,1> head_fc1_bias_t;
typedef ap_uint<1> layer60_index;
typedef ap_ufixed<16,9,AP_RND_CONV,AP_SAT,0> head_act_t;
typedef ap_ufixed<2,32> head_act_param_t;
typedef ap_fixed<18,8> head_act_table_t;
typedef ap_fixed<22,15> head_fc2_accum_t;
typedef ap_fixed<16,9,AP_RND_CONV,AP_SAT,0> head_fc2_t;
typedef ap_fixed<2,2> head_fc2_weight_t;
typedef ap_ufixed<2,32> head_fc2_bias_t;
typedef ap_uint<1> layer63_index;
typedef ap_fixed<21,5> result_t;
typedef ap_ufixed<4,-4> head_fc2_affine_scale_t;
typedef ap_fixed<15,-1> head_fc2_affine_bias_t;

// hls-fpga-machine-learning insert emulator-defines


#endif
