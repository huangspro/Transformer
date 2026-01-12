#include "../include/WordList.h"
#include "../include/Embedding.h"
#include "../include/Attention.h"
#include "../include/Parameters.h"

using namespace std;

int main() {
    
    //wordlist
    vector<string> w={"a","b","c","d","e","f","g","h","i","j"};
    WordList* W = new WordList(w);
    //embedding
    Embedding E(W);
    //attention
    Attention A;
    
    vector<string> in={"a","b","c","d","e","a"};
    E.load_input(in);
    E.PositionalEncoding();
    E.output_with_posi();
    
    A.load_input(E.output);
    A.cal_output();
    cout<<A.output.n_rows<<" "<<A.output.n_cols;
    A.output.print();
}
