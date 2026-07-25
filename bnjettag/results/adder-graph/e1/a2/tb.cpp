#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "myproject.h"
extern const double W_REF[N_OUT][N_IN];
extern const double B_REF[N_OUT];
int main() {
  srand(20260724);
  int bad = 0;
  for (int trial = 0; trial < 256; trial++) {
    input_t in[N_TOK][N_IN];
    double din[N_TOK][N_IN];
    for (int t = 0; t < N_TOK; t++)
      for (int i = 0; i < N_IN; i++) {
        int q = (rand() % 256) - 128;          // full ap_fixed<8,4> grid
        din[t][i] = q / 16.0;
        in[t][i] = (input_t)din[t][i];
      }
    result_t out[N_TOK][N_OUT];
    myproject(in, out);
    for (int t = 0; t < N_TOK; t++)
      for (int o = 0; o < N_OUT; o++) {
        double ref = B_REF[o];
        for (int i = 0; i < N_IN; i++) ref += W_REF[o][i] * din[t][i];
        if (std::fabs(out[t][o].to_double() - ref) > 1e-12) {
          if (bad < 5) printf("MISMATCH t=%d o=%d got=%.10f ref=%.10f\n",
                              t, o, out[t][o].to_double(), ref);
          bad++;
        }
      }
  }
  printf(bad ? "FAIL %d mismatches\n" : "EXACT_MATCH\n", bad);
  return bad ? 1 : 0;
}
