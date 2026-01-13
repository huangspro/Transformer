#ifndef _ATTENTION_H_
#define _ATTENTION_H_

#include "Parameters.h"

class Attention{
public:
  //respectively for generating Query, Key, Value, storing output and output, and joining all the output 
  std::vector<arma::mat> q_gen, k_gen, v_gen;
  arma::mat input, output_join,output;
  
  Attention();
  void load_input(arma::mat); //load input data from the previous embedding layer
  void cal_output();  //compute the output matrix
};

#endif
