#include "../include/WordList.h"
#include "../include/Embedding.h"

#include <armadillo>
#include <iostream>
#include<vector>
#include<string>

using namespace std;

int main() {
    
    
    vector<string> w={"a","b","c","d","e","f","g","h","i","j"};
    WordList* W = new WordList(w);
    Embedding E(W);
    vector<string> in={"a","b","c","d","e","a"};
    
    E.load_input(in);
    E.PositionalEncoding();
    E.output_with_posi();
    E.output.print();
}
