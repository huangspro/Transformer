#ifndef _ATTENTION_H_
#define _ATTENTION_H_

#include "Parameters.h"

class Attention{
public:
  //5 matrix respectively for generating Query, Key, Value, and storing output and output
  arma::mat input, q_gen, k_gen, v_gen, output;
  
  Attention();
  void load_input(arma::mat); //load input data from the previous embedding layer
  void cal_output();  //compute the output matrix
};

#endif
