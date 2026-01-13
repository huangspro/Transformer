#include "../include/Attention.h"

Attention::Attention(bool m):is_masked(m){
  for(int i=0;i<Attention_Head;i++){
    q_gen.push_back(arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head));
    k_gen.push_back(arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head));
    v_gen.push_back(arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth/Attention_Head));
  }
  output_join = arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth);
}

void Attention::load_input(arma::mat I){
  input = I;
  output = arma::randn<arma::mat>(input.n_rows, 0);
}

//this function is for computing the sofemax of a vecgtor, for row vector
void Softmax(arma::mat& in){
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

void Attention::cal_output(){
  //prepare
  std::vector<arma::mat> q,k,v,tem,tem_output;
  std::vector<double> dimension_sqrt;
  for(int i=0;i<Attention_Head;i++){
    tem.push_back(arma::mat(input.n_rows, input.n_rows));  
    q.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    k.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    v.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    tem_output.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    dimension_sqrt.push_back(std::sqrt(Embedding_Depth/Attention_Head));  //prepare the sqrt of dimension of querys and keys
  }
  
  for(int i=0;i<Attention_Head;i++){
    //compute the q,k,v
    q[i] = input * q_gen[i];
    k[i] = input * k_gen[i];
    v[i] = input * v_gen[i];
  
    //compute q*k^T
    tem[i] = q[i]*k[i].t()/dimension_sqrt[i];
    
    //compute masked
    if(is_masked){
      //noted that tem's shape is : input_row * input_row
      for(int i=0;i<input.n_rows;i++){
        for(int j=0;j<input.n_rows;j++){
           if(j>i)tem[i](i,j)=-arma::datum::inf;
        }
      }
    }
    
    //implement softmax function
    Softmax(tem[i]);
    
    //output
    tem_output[i] = tem[i]*v[i];
    
    //join the output
    output = arma::join_horiz(output,tem_output[i]);
  }
  output=output*output_join;
}
