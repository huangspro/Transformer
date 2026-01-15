#include "../include/WordList.h"
#include "../include/Embedding.h"
#include "../include/Attention.h"
#include "../include/Parameters.h"
#include "../include/Normalization.h"
#include "../include/FFN.h"
#include "../include/Output.h"

using namespace std;

int main() {
    //set ramdom seed
    arma::arma_rng::set_seed_random();
    
    //wordlist
    vector<string> w={"a","b","c","d","e","f","g","h","i","j"}; //10 words
    WordList* W = new WordList(w);
    //embedding
    Embedding E_en(W);
    Embedding E_de(W);
    //attention
    Attention A_en(false, 1);
    Attention A_de(true, 1);
    Attention A(false, 2);
    //Normalizetion
    Normalization N_en_1,N_de_1,N_en_2,N_de_2;
    //ffn
    FFN F_en,F_de;
    //output
    Output O;
    
    vector<string> en_in={"a","b","c"};
    vector<string> de_in={"d","e","f"};
    E_en.load_input(en_in);
    E_en.load_input(de_in);
    
    E_en.PositionalEncoding();
    E_en.output_with_posi();
    E_de.PositionalEncoding();
    E_de.output_with_posi();
    
    A_en.load_input(E_en.output);
    A_en.cal_output();
    A_de.load_input(E_de.output);
    A_de.cal_output();
    
    N_en_1.load_input(A_en.output,E_en.output);
    N_en_1.cal_output();
    N_de_1.load_input(A_de.output,E_de.output);
    N_en_1.cal_output();
    
    F_en.cal_output();
    F_en.load_input(N_en_1.output);
    F_de.cal_output();
    F_de.load_input(N_de_1.output);
    
    N_en_2.load_input(F_en.output,N_en_1.output);
    N_en_2.cal_output();
    N_de_2.load_input(F_de.output,N_de_1.output);
    N_en_2.cal_output();
    
    A.load_input(N_en_2.output, N_de_2.output);
    A.cal_output();
   
    O.load_input(A.output);
    O.cal_output();
    
    O.output.print();
}
