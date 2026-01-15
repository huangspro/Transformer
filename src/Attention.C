#include "../include/Attention.h"

Attention::Attention(bool m, int mode):is_masked(m), mode(mode){
  for(int i=0;i<Attention_Head;i++){
    q_gen.push_back(arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth));
    k_gen.push_back(arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth));
    v_gen.push_back(arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth));
  }
  output_join = arma::randn<arma::mat>(Embedding_Depth, Embedding_Depth);
}

void Attention::load_input(arma::mat E, arma::mat D){
  EN=E;
  DE=D;
  output = arma::randn<arma::mat>(input.n_rows, 0);
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
  std::vector<arma::mat>q,k,v,tem,tem_output;
  std::vector<double> dimension_sqrt;
  for(int i=0;i<Attention_Head;i++){
    tem.push_back(arma::mat(input.n_rows, input.n_rows));  
    tem_output.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    q.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    k.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    v.push_back(arma::mat(input.n_rows, Embedding_Depth/Attention_Head));
    dimension_sqrt.push_back(std::sqrt(Embedding_Depth/Attention_Head));  //prepare the sqrt of dimension of querys and keys
  }
  
  for(int i=0;i<Attention_Head;i++){
    if(mode==2){
      //compute the q,k,v
      q[i] = EN * q_gen[i];
      k[i] = EN * k_gen[i];
      v[i] = DE * v_gen[i];
    }else{
      std::cout<<input.n_rows<<" "<<input.n_cols<<q_gen[i].n_rows<<" "<<q_gen[i].n_cols<<std::flush<<std::endl;;
      q[i] = input * q_gen[i];
      k[i] = input * k_gen[i];
      v[i] = input * v_gen[i];
    }
    //compute q*k^T
    tem[i] = q[i]*k[i].t()/dimension_sqrt[i];
    
    //compute masked
    if(is_masked){
      //noted that tem's shape is : input_row * input_row
      for(int ii=0;ii<input.n_rows;ii++){
        for(int j=0;j<input.n_rows;j++){
           if(j>ii)tem[i](ii,j)=-arma::datum::inf;
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
