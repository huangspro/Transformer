#include "../include/FFN.h"

FFN::FFN(){}

void FFN::load_input(arma::mat in){
  //the shape of W1 is Embedding_Depth*FFN_Depth
  //the shape of W2 is FFN_Depth*Embedding_Depth
  //using variance of sqrt(2/in) to initialize the weight matrix
  W1 = std::sqrt(2/(in.n_rows*in.n_cols))*arma::randn(Embedding_Depth, FFN_Depth);
  W2 = std::sqrt(2/(in.n_rows*in.n_cols))*arma::randn(FFN_Depth, Embedding_Depth);
  
  b1 = arma::randn(in.n_rows, FFN_Depth);
  b2 = arma::randn(in.n_rows, Embedding_Depth);
  
  output = arma::mat(in.n_rows, Embedding_Depth);
  input = in;
}

void FFN::cal_output(){
  //implement relu function
  arma::mat tem = input*W1+b1;
  for(int i=0;i<tem.n_rows;i++){
    for(int ii=0;ii<tem.n_rows;ii++){
      tem(i,ii)=tem(i,ii)<0?0:tem(i,ii);
    }
  }
  output = tem*W2+b2;  
}
