#include "../include/Output.h"

Output::Output(){}

void Output::load_input(arma::mat in){
  input = in;
  W = arma::randn(Embedding_Depth, WordList_Length);
  B = arma::randn(in.n_rows, WordList_Length);
}

//this function is for computing the sofemax of a vecgtor, for row vector
void Output::Softmax(arma::mat& in){
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

void Output::cal_output(){
  output=input * W + B;
  Softmax(output);
}


