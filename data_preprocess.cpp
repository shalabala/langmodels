#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

int main() {
  unordered_map<string, int> vocab_to_indices;
  string line;
  ifstream train_data_file("train.txt");
  ofstream tokensfile("tokens_binary.txt");
  ofstream vocabfile("vocab_2.txt");

  vector<int> tokens;
  int no_of_tokens = 0;
  int lines_read = 0;
  while (getline(train_data_file, line)) {
    lines_read++;
    size_t pos = 0, found;
    int current_token_index;
    while ((found = line.find(' ', pos)) != string::npos) {
      string token = line.substr(pos, found - pos);
      auto current_iterator = vocab_to_indices.find(token);
      if (current_iterator == vocab_to_indices.end()) {
        current_token_index = no_of_tokens;
        vocab_to_indices[token] = no_of_tokens;
        no_of_tokens++;
      } else {
        current_token_index = current_iterator->second;
      }

      tokens.push_back(current_token_index);
      pos = found + 1;
    }
    if (lines_read % 50000 == 0) {
      cout << "Line " << lines_read << " Complete: " << lines_read / 74004228.0
           << endl;
    }
  }
  train_data_file.close();
  
  cout << "Reading phase finished, number of tokens read: " << tokens.size()
       << ", vocabulary size: " << vocab_to_indices.size() << endl;

  int written_tokens = 0;
  for (auto &token : tokens) {
    ++written_tokens;
    tokensfile << token << ";";
    if (written_tokens % 50000 == 0) {
      cout << "Tokens written: " << written_tokens
           << " Complete: " << written_tokens / ((double)tokens.size()) << endl;
    }
  }

  tokensfile.close();
  
  int written_vocabs = 0;
  for (auto &vocabidx : vocab_to_indices) {
    ++written_vocabs;
    vocabfile << vocabidx.first << ":" << vocabidx.second << endl;
    if (written_tokens % 50000 == 0) {
      cout << "Vocabs written: " << written_vocabs
           << " Complete: " << written_vocabs / ((double)vocab_to_indices.size()) << endl;
    }
  }
  vocabfile.close();
 
}