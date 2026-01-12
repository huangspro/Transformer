/*
This file defines a wordlist class which is used for embedding
noted that this class can be trained
*/

#ifndef _WORDLIST_H_
#define _WORDLIST_H_

#include "Parameters.h"

class WordList{
public:
  int word_number;  
  arma::mat V; //all the wordlist, shape: ---> word_number * Embedding_Depth
  std::vector<std::string> W;  //the real string of a word
  
  WordList(std::vector<std::string>&);  //initialize the matrix into E and copy input into W
  
  arma::rowvec getRow(int);  //get a row vector, 0-based
  arma::colvec getCol(int);  //get a col vector, 0-based
  
  arma::rowvec getIndex(std::string);  //get the vector of a word
  
  bool save();  //this funciton can write the wordlist into a binary file
  bool read();  //this function can read a wordlist from a binary file
};

#endif
