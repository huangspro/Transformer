#include "../include/Embedding.h"
#include<vector>
#include<string>
#include<cmath>

Embedding::Embedding(int l):input_length(l){
  positional = output = input = arma::mat(input_length, Embedding_Depth);
}

arma::mat Embedding::load_input(std::vector<std::string>& I){
  for(int i=0;i<I.size();i++){
    input.row(i)=w->getIndex(I[i]);
  }
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
