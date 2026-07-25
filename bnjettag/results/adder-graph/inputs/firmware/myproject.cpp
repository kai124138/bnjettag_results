#include <iostream>

#include "myproject.h"
#include "parameters.h"


void myproject(
    input_1_t input_1[10*16],
    result_t layer65_out[5]
) {

    // hls-fpga-machine-learning insert IO
    #pragma HLS ARRAY_RESHAPE variable=input_1 complete dim=0
    #pragma HLS ARRAY_PARTITION variable=layer65_out complete dim=0
    #pragma HLS INTERFACE ap_vld port=input_1,layer65_out 
    #pragma HLS PIPELINE

    // hls-fpga-machine-learning insert load weights
#ifndef __SYNTHESIS__
    static bool loaded_weights = false;
    if (!loaded_weights) {
        nnet::load_weights_from_txt<input_proj_weight_t, 512>(w3, "w3.txt");
        nnet::load_weights_from_txt<input_proj_bias_t, 320>(b3, "b3.txt");
        nnet::load_weights_from_txt<input_proj_affine_scale_t, 32>(s5, "s5.txt");
        nnet::load_weights_from_txt<input_proj_affine_bias_t, 32>(b5, "b5.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w7, "w7.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b7, "b7.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w9, "w9.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b9, "b9.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w15, "w15.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b15, "b15.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w20, "w20.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b20, "b20.txt");
        nnet::load_weights_from_txt<bit_block_0_attn_Wo_affine_scale_t, 32>(s22, "s22.txt");
        nnet::load_weights_from_txt<bit_block_0_attn_Wo_affine_bias_t, 32>(b22, "b22.txt");
        nnet::load_weights_from_txt<bit_block_0_ffn_fc1_weight_t, 2048>(w25, "w25.txt");
        nnet::load_weights_from_txt<bit_block_0_ffn_fc1_bias_t, 640>(b25, "b25.txt");
        nnet::load_weights_from_txt<bit_block_0_ffn_fc2_weight_t, 2048>(w28, "w28.txt");
        nnet::load_weights_from_txt<bit_block_0_ffn_fc2_bias_t, 320>(b28, "b28.txt");
        nnet::load_weights_from_txt<bit_block_0_ffn_fc2_affine_scale_t, 32>(s30, "s30.txt");
        nnet::load_weights_from_txt<bit_block_0_ffn_fc2_affine_bias_t, 32>(b30, "b30.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w33, "w33.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b33, "b33.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w35, "w35.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b35, "b35.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w41, "w41.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b41, "b41.txt");
        nnet::load_weights_from_txt<model_default_t, 1024>(w46, "w46.txt");
        nnet::load_weights_from_txt<model_default_t, 320>(b46, "b46.txt");
        nnet::load_weights_from_txt<bit_block_1_attn_Wo_affine_scale_t, 32>(s48, "s48.txt");
        nnet::load_weights_from_txt<bit_block_1_attn_Wo_affine_bias_t, 32>(b48, "b48.txt");
        nnet::load_weights_from_txt<bit_block_1_ffn_fc1_weight_t, 2048>(w51, "w51.txt");
        nnet::load_weights_from_txt<bit_block_1_ffn_fc1_bias_t, 640>(b51, "b51.txt");
        nnet::load_weights_from_txt<bit_block_1_ffn_fc2_weight_t, 2048>(w54, "w54.txt");
        nnet::load_weights_from_txt<bit_block_1_ffn_fc2_bias_t, 320>(b54, "b54.txt");
        nnet::load_weights_from_txt<bit_block_1_ffn_fc2_affine_scale_t, 32>(s56, "s56.txt");
        nnet::load_weights_from_txt<bit_block_1_ffn_fc2_affine_bias_t, 32>(b56, "b56.txt");
        nnet::load_weights_from_txt<head_fc1_weight_t, 1024>(w60, "w60.txt");
        nnet::load_weights_from_txt<head_fc1_bias_t, 32>(b60, "b60.txt");
        nnet::load_weights_from_txt<head_fc2_weight_t, 160>(w63, "w63.txt");
        nnet::load_weights_from_txt<head_fc2_bias_t, 5>(b63, "b63.txt");
        nnet::load_weights_from_txt<head_fc2_affine_scale_t, 5>(s65, "s65.txt");
        nnet::load_weights_from_txt<head_fc2_affine_bias_t, 5>(b65, "b65.txt");
        loaded_weights = true;    }
#endif
    // ****************************************
    // NETWORK INSTANTIATION
    // ****************************************

    // hls-fpga-machine-learning insert layers

    input_proj_t layer3_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer3_out complete dim=0

    input_proj_affine_t layer5_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer5_out complete dim=0

    bit_block_0_attn_Wq_iq_t layer6_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer6_out complete dim=0

    bit_block_0_attn_Wq_t layer7_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer7_out complete dim=0

    bit_block_0_attn_Wk_iq_t layer8_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer8_out complete dim=0

    bit_block_0_attn_Wk_t layer9_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer9_out complete dim=0

    bit_block_0_attn_scores_t layer12_out[4*10*10];
    #pragma HLS ARRAY_PARTITION variable=layer12_out complete dim=0

    bit_block_0_attn_softmax_t layer13_out[4*10*10];
    #pragma HLS ARRAY_PARTITION variable=layer13_out complete dim=0

    bit_block_0_attn_Wv_iq_t layer14_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer14_out complete dim=0

    bit_block_0_attn_Wv_t layer15_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer15_out complete dim=0

    bit_block_0_attn_ctx_t layer18_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer18_out complete dim=0

    bit_block_0_attn_Wo_t layer20_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer20_out complete dim=0

    bit_block_0_attn_Wo_affine_t layer22_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer22_out complete dim=0

    bit_block_0_add_attn_t layer23_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer23_out complete dim=0

    bit_block_0_ffn_fc1_iq_t layer24_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer24_out complete dim=0

    bit_block_0_ffn_fc1_t layer25_out[10*64];
    #pragma HLS ARRAY_PARTITION variable=layer25_out complete dim=0

    bit_block_0_ffn_act_t layer26_out[10*64];
    #pragma HLS ARRAY_PARTITION variable=layer26_out complete dim=0

    bit_block_0_ffn_fc2_t layer28_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer28_out complete dim=0

    bit_block_0_ffn_fc2_affine_t layer30_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer30_out complete dim=0

    bit_block_0_add_ffn_t layer31_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer31_out complete dim=0

    bit_block_1_attn_Wq_iq_t layer32_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer32_out complete dim=0

    bit_block_1_attn_Wq_t layer33_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer33_out complete dim=0

    bit_block_1_attn_Wk_iq_t layer34_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer34_out complete dim=0

    bit_block_1_attn_Wk_t layer35_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer35_out complete dim=0

    bit_block_1_attn_scores_t layer38_out[4*10*10];
    #pragma HLS ARRAY_PARTITION variable=layer38_out complete dim=0

    bit_block_1_attn_softmax_t layer39_out[4*10*10];
    #pragma HLS ARRAY_PARTITION variable=layer39_out complete dim=0

    bit_block_1_attn_Wv_iq_t layer40_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer40_out complete dim=0

    bit_block_1_attn_Wv_t layer41_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer41_out complete dim=0

    bit_block_1_attn_ctx_t layer44_out[10*4*8];
    #pragma HLS ARRAY_PARTITION variable=layer44_out complete dim=0

    bit_block_1_attn_Wo_t layer46_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer46_out complete dim=0

    bit_block_1_attn_Wo_affine_t layer48_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer48_out complete dim=0

    bit_block_1_add_attn_t layer49_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer49_out complete dim=0

    bit_block_1_ffn_fc1_iq_t layer50_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer50_out complete dim=0

    bit_block_1_ffn_fc1_t layer51_out[10*64];
    #pragma HLS ARRAY_PARTITION variable=layer51_out complete dim=0

    bit_block_1_ffn_act_t layer52_out[10*64];
    #pragma HLS ARRAY_PARTITION variable=layer52_out complete dim=0

    bit_block_1_ffn_fc2_t layer54_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer54_out complete dim=0

    bit_block_1_ffn_fc2_affine_t layer56_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer56_out complete dim=0

    bit_block_1_add_ffn_t layer57_out[10*32];
    #pragma HLS ARRAY_PARTITION variable=layer57_out complete dim=0

    gap_t layer58_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer58_out complete dim=0

    head_fc1_t layer60_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer60_out complete dim=0

    head_act_t layer61_out[32];
    #pragma HLS ARRAY_PARTITION variable=layer61_out complete dim=0

    head_fc2_t layer63_out[5];
    #pragma HLS ARRAY_PARTITION variable=layer63_out complete dim=0

    nnet::einsum_dense<input_1_t, input_proj_t, config3>(input_1, layer3_out, w3, b3); // input_proj

    nnet::normalize<input_proj_t, input_proj_affine_t, config5>(layer3_out, layer5_out, s5, b5); // input_proj_affine

    nnet::bit_block_0_attn_Wq_iq<input_proj_affine_t, bit_block_0_attn_Wq_iq_t>(layer5_out, layer6_out); // bit_block_0_attn_Wq_iq

    nnet::einsum_dense<bit_block_0_attn_Wq_iq_t, bit_block_0_attn_Wq_t, config7>(layer6_out, layer7_out, w7, b7); // bit_block_0_attn_Wq

    nnet::bit_block_0_attn_Wk_iq<input_proj_affine_t, bit_block_0_attn_Wk_iq_t>(layer5_out, layer8_out); // bit_block_0_attn_Wk_iq

    nnet::einsum_dense<bit_block_0_attn_Wk_iq_t, bit_block_0_attn_Wk_t, config9>(layer8_out, layer9_out, w9, b9); // bit_block_0_attn_Wk

    nnet::einsum<bit_block_0_attn_Wq_t, bit_block_0_attn_Wk_t, bit_block_0_attn_scores_t, config12>(layer7_out, layer9_out, layer12_out); // bit_block_0_attn_scores

    nnet::softmax_multidim<bit_block_0_attn_scores_t, bit_block_0_attn_softmax_t, softmax_config13>(layer12_out, layer13_out); // bit_block_0_attn_softmax

    nnet::bit_block_0_attn_Wv_iq<input_proj_affine_t, bit_block_0_attn_Wv_iq_t>(layer5_out, layer14_out); // bit_block_0_attn_Wv_iq

    nnet::einsum_dense<bit_block_0_attn_Wv_iq_t, bit_block_0_attn_Wv_t, config15>(layer14_out, layer15_out, w15, b15); // bit_block_0_attn_Wv

    nnet::einsum<bit_block_0_attn_softmax_t, bit_block_0_attn_Wv_t, bit_block_0_attn_ctx_t, config18>(layer13_out, layer15_out, layer18_out); // bit_block_0_attn_ctx

    nnet::einsum_dense<bit_block_0_attn_ctx_t, bit_block_0_attn_Wo_t, config20>(layer18_out, layer20_out, w20, b20); // bit_block_0_attn_Wo

    nnet::normalize<bit_block_0_attn_Wo_t, bit_block_0_attn_Wo_affine_t, config22>(layer20_out, layer22_out, s22, b22); // bit_block_0_attn_Wo_affine

    nnet::add<input_proj_affine_t, bit_block_0_attn_Wo_affine_t, bit_block_0_add_attn_t, config23>(layer5_out, layer22_out, layer23_out); // bit_block_0_add_attn

    nnet::bit_block_0_ffn_fc1_iq<bit_block_0_add_attn_t, bit_block_0_ffn_fc1_iq_t>(layer23_out, layer24_out); // bit_block_0_ffn_fc1_iq

    nnet::einsum_dense<bit_block_0_ffn_fc1_iq_t, bit_block_0_ffn_fc1_t, config25>(layer24_out, layer25_out, w25, b25); // bit_block_0_ffn_fc1

    nnet::thresholded_relu<bit_block_0_ffn_fc1_t, bit_block_0_ffn_act_param_t, bit_block_0_ffn_act_t, thresholdedrelu_config26>(layer25_out, 0.0, layer26_out); // bit_block_0_ffn_act

    nnet::einsum_dense<bit_block_0_ffn_act_t, bit_block_0_ffn_fc2_t, config28>(layer26_out, layer28_out, w28, b28); // bit_block_0_ffn_fc2

    nnet::normalize<bit_block_0_ffn_fc2_t, bit_block_0_ffn_fc2_affine_t, config30>(layer28_out, layer30_out, s30, b30); // bit_block_0_ffn_fc2_affine

    nnet::add<bit_block_0_add_attn_t, bit_block_0_ffn_fc2_affine_t, bit_block_0_add_ffn_t, config31>(layer23_out, layer30_out, layer31_out); // bit_block_0_add_ffn

    nnet::bit_block_1_attn_Wq_iq<bit_block_0_add_ffn_t, bit_block_1_attn_Wq_iq_t>(layer31_out, layer32_out); // bit_block_1_attn_Wq_iq

    nnet::einsum_dense<bit_block_1_attn_Wq_iq_t, bit_block_1_attn_Wq_t, config33>(layer32_out, layer33_out, w33, b33); // bit_block_1_attn_Wq

    nnet::bit_block_1_attn_Wk_iq<bit_block_0_add_ffn_t, bit_block_1_attn_Wk_iq_t>(layer31_out, layer34_out); // bit_block_1_attn_Wk_iq

    nnet::einsum_dense<bit_block_1_attn_Wk_iq_t, bit_block_1_attn_Wk_t, config35>(layer34_out, layer35_out, w35, b35); // bit_block_1_attn_Wk

    nnet::einsum<bit_block_1_attn_Wq_t, bit_block_1_attn_Wk_t, bit_block_1_attn_scores_t, config38>(layer33_out, layer35_out, layer38_out); // bit_block_1_attn_scores

    nnet::softmax_multidim<bit_block_1_attn_scores_t, bit_block_1_attn_softmax_t, softmax_config39>(layer38_out, layer39_out); // bit_block_1_attn_softmax

    nnet::bit_block_1_attn_Wv_iq<bit_block_0_add_ffn_t, bit_block_1_attn_Wv_iq_t>(layer31_out, layer40_out); // bit_block_1_attn_Wv_iq

    nnet::einsum_dense<bit_block_1_attn_Wv_iq_t, bit_block_1_attn_Wv_t, config41>(layer40_out, layer41_out, w41, b41); // bit_block_1_attn_Wv

    nnet::einsum<bit_block_1_attn_softmax_t, bit_block_1_attn_Wv_t, bit_block_1_attn_ctx_t, config44>(layer39_out, layer41_out, layer44_out); // bit_block_1_attn_ctx

    nnet::einsum_dense<bit_block_1_attn_ctx_t, bit_block_1_attn_Wo_t, config46>(layer44_out, layer46_out, w46, b46); // bit_block_1_attn_Wo

    nnet::normalize<bit_block_1_attn_Wo_t, bit_block_1_attn_Wo_affine_t, config48>(layer46_out, layer48_out, s48, b48); // bit_block_1_attn_Wo_affine

    nnet::add<bit_block_0_add_ffn_t, bit_block_1_attn_Wo_affine_t, bit_block_1_add_attn_t, config49>(layer31_out, layer48_out, layer49_out); // bit_block_1_add_attn

    nnet::bit_block_1_ffn_fc1_iq<bit_block_1_add_attn_t, bit_block_1_ffn_fc1_iq_t>(layer49_out, layer50_out); // bit_block_1_ffn_fc1_iq

    nnet::einsum_dense<bit_block_1_ffn_fc1_iq_t, bit_block_1_ffn_fc1_t, config51>(layer50_out, layer51_out, w51, b51); // bit_block_1_ffn_fc1

    nnet::thresholded_relu<bit_block_1_ffn_fc1_t, bit_block_1_ffn_act_param_t, bit_block_1_ffn_act_t, thresholdedrelu_config52>(layer51_out, 0.0, layer52_out); // bit_block_1_ffn_act

    nnet::einsum_dense<bit_block_1_ffn_act_t, bit_block_1_ffn_fc2_t, config54>(layer52_out, layer54_out, w54, b54); // bit_block_1_ffn_fc2

    nnet::normalize<bit_block_1_ffn_fc2_t, bit_block_1_ffn_fc2_affine_t, config56>(layer54_out, layer56_out, s56, b56); // bit_block_1_ffn_fc2_affine

    nnet::add<bit_block_1_add_attn_t, bit_block_1_ffn_fc2_affine_t, bit_block_1_add_ffn_t, config57>(layer49_out, layer56_out, layer57_out); // bit_block_1_add_ffn

    nnet::global_pooling1d_cl<bit_block_1_add_ffn_t, gap_t, config58>(layer57_out, layer58_out); // gap

    nnet::dense<gap_t, head_fc1_t, config60>(layer58_out, layer60_out, w60, b60); // head_fc1

    nnet::thresholded_relu<head_fc1_t, head_act_param_t, head_act_t, thresholdedrelu_config61>(layer60_out, 0.0, layer61_out); // head_act

    nnet::dense<head_act_t, head_fc2_t, config63>(layer61_out, layer63_out, w63, b63); // head_fc2

    nnet::normalize<head_fc2_t, result_t, config65>(layer63_out, layer65_out, s65, b65); // head_fc2_affine

}

