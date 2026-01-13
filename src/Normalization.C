#include "../include/Normalization.h"

Normalization::Normalization(){}

void Normalization::load_input(arma::mat x, arma::mat subx){
  (*this).x=x;
  (*this).subx=subx;
}

void Normalization::cal_output(){
  output = x+subx;
  //output.print();
  std::cout<<"ppp";
  for(int i=0; i<output.n_rows; i++){
    arma::rowvec row = output.row(i);
    double mean = arma::mean(row);          // mean
    double variance = arma::var(row,1);     // variance with n
    row = (row - mean) / std::sqrt(variance + 1e-5);  // normalize
    output.row(i) = row;
  }
  //output.print();
}
