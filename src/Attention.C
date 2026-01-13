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
//this function is for computing the sofemax of a vecgtor, for row vector
void Softmax(arma::mat& in){
  for(int ii=0;ii<in.n_rows;ii++){
    double tem = 0;
    for(int i=0;i<in.n_cols;i++){
      tem+=std::pow(2.7183,in(ii,i));
    }
    for(int i=0;i<in.n_cols;i++){
      in(ii,i)=std::pow(2.7183,in(ii,i))/tem;
    }
  }
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
  Softmax(tem);
  tem.print();
  //output
  output = tem*v;
}



