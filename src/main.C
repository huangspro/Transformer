#include "../include/WordList.h"
#include "../include/Embedding.h"
#include "../include/Attention.h"
#include "../include/Parameters.h"
#include "../include/Normalization.h"
#include "../include/FFN.h"

using namespace std;

int main() {
    
    //wordlist
    vector<string> w={"a","b","c","d","e","f","g","h","i","j"};
    WordList* W = new WordList(w);
    //embedding
    Embedding E(W);
    //attention
    Attention A(false);
    Normalization N;
    
    vector<string> in={"a","b","c"};
    E.load_input(in);
    E.PositionalEncoding();
    E.output_with_posi();
    
    A.load_input(E.output);
    A.cal_output();
    A.output.print();
    
    N.load_input(E.output, A.output);
    N.cal_output();
    N.output.print();
    
    FFN FF;
    FF.load_input(N.output);
    FF.cal_output();
    FF.output.print();
}
