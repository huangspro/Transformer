#include "../include/Embedding.h"
#include<vector>
#include<string>
#include<cmath>
#include<iostream>
Embedding::Embedding(WordList* w):input_length(0),w(w){
  input = arma::mat(0,0);
  output = arma::mat(0,0);
  positional = arma::mat(0,0);
}

arma::mat Embedding::load_input(std::vector<std::string>& I){
  input_length=I.size();
  input = arma::mat(input_length, Embedding_Depth);
  positional = arma::mat(input_length, Embedding_Depth);
  output = arma::mat(input_length, Embedding_Depth);
  //for(int ii=0;ii<input_length;ii++){
    std::cout<<input_length<<std::flush;
    //input.row(i)=w->getIndex(I[i]);
  //}
}

void Embedding::PositionalEncoding(){
  for(double i=0;i<input_length;i++){
    for(double j=0;j<(Embedding_Depth-1)/2;j++){  //from this line we indicate that Embedding_Depth should be even number.
      positional(i, 2*j) = std::sin(i/std::pow(1000,2*j/Embedding_Depth));
      positional(i, 2*j+1) = std::cos(i/std::pow(1000,2*j/Embedding_Depth));
    }
  }
}

void Embedding::output_with_posi(){
  output = input + positional;
}
