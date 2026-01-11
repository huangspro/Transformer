#include "../include/WordList.h"
#include<fstream>

WordList::WordList(std::vector<std::string>& input):word_number(input.size()){
  V = arma::eye<arma::mat>(word_number, Embedding_Depth); //initialize the matrix
  for(int i=0;i<input.size();i++){  //copy the word
    W.push_back(input[i]);
  }
}

arma::rowvec WordList::getRow(int i){
  return V.row(i);
}

arma::colvec WordList::getCol(int i){
  return V.col(i);
}

bool WordList::save() {
  std::ofstream out("WordList_Save", std::ios::binary);
  if (!out) false;
  out.write(reinterpret_cast<const char*>(this), sizeof(*this));
  out.close();
  return true;
}

bool WordList::read() {
  std::ifstream in("WordList_Save", std::ios::binary);
  if (!in) return false;
  in.read(reinterpret_cast<char*>(this), sizeof(*this));
  in.close();
  return true;
}
