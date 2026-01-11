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
  WordList 
  Embedding(int);
  aram::mat load_input(arma::vec);  //read the input data and output the matrix
}

#endif
