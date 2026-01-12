#include "Attention.h"

Attention::Attention(){
  output = arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head);
  q_gen = arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head);
  k_gen = arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head);
  v_gen = arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head);
}

void Attention::load_input(arma::mat I){
  input = I;
}

void Attention::cal_output(){
  arma::mat tem = arma::mat(input.n_rows, input.n_rows);  
  arma::mat q = arma::mat(input.n_rows, Embedding_Depth/Attention_Head);
  arma::mat k = arma::mat(input.n_rows, Embedding_Depth/Attention_Head);
  arma::mat v = arma::mat(input.n_rows, Embedding_Depth/Attention_Head);
  double dimension_sqrt = std::sqrt(Embedding_Depth/Attention_Head);  //prepare the sqrt of dimension of querys and keys
  
  //compute the q,k,v
  q = input * q_gen;
  k = input * k_gen;
  v = input * v_gen;
  
  //compute q*k^T
  tem = q*k.t()/dimension_sqrt;
  //output
  output = tem*v;
}
