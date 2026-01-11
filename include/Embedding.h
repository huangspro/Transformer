/*
This file defines the embedding layer in transformer
*/

#ifndef _EMBEDDING_H_
#define _EMBEDDING_H_

#include "Parameters.h"
#include "WordList.h"

class Embedding{
public:
  int input_length;
  arma::mat input,output,positional;
  WordList* w;  //use the wordlist from outside, and pass the gradient to the wordlist
  
  Embedding(int);
  arma::mat load_input(std::vector<std::string>&);  //read the input data and output the matrix
  void PositionalEncoding();  //implement positional encoding for the matrix and return the result matrix, shaping the same
  void output_with_posi();  //assign the output with the addition of positional encoding matrix and embedding matrix
};

#endif
