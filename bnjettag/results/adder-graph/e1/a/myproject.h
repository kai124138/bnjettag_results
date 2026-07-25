#include "ap_fixed.h"
typedef ap_fixed<8,4> input_t;
typedef ap_fixed<26,10> acc_t;
typedef ap_fixed<26,10> result_t;
typedef ap_fixed<18,2> bias_t;
typedef ap_fixed<10,6> gsum_t;
#define N_TOK 10
#define N_IN 32
#define N_OUT 64
void myproject(input_t in[N_TOK][N_IN], result_t out[N_TOK][N_OUT]);
