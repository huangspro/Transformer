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
  arma::mat tem = arma::mat(Embedding_Depth, Embedding_Depth/Attention_Head);  
  arma::mat q = arma::mat(Embedding_Depth, Embedding_Depth/Attention_Head);
  arma::mat k = arma::mat(Embedding_Depth, Embedding_Depth/Attention_Head);
  arma::mat v = arma::mat(Embedding_Depth, Embedding_Depth/Attention_Head);
  double dimention = std::sqrt(Embedding_Depth/Attention_Head);  //prepare the sqrt of dimention of querys and keys
  
  //compute the q,k,v
  q = input * q_gen;
  k = input * k_gen;
  v = input * v_gen;
  
  //compute q*k^T
  
  
}
