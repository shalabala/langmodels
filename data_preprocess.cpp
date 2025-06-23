#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#define THRESHOLD 32767
using token = uint16_t;
using namespace std;

struct vocab_entry {
  token id = 0;
  uint32_t occurrences = 0;
  vocab_entry() = default;
  vocab_entry(uint16_t id, uint32_t occurrences)
      : id(id), occurrences(occurrences) {}
};

// 1. read the tokens
void read_tokens(unordered_map<string, vocab_entry *> &vocab_entries,
                 vector<vocab_entry *> &tokens,
                 string datafile) {
  ifstream train_data_file(datafile);
  int no_of_tokens = 0;
  int lines_read = 0;
  string line;
  while (getline(train_data_file, line)) {
    lines_read++;
    size_t pos = 0, found;
    while ((found = line.find(' ', pos)) != string::npos) {
      string token = line.substr(pos, found - pos);
      vocab_entry *current;
      auto current_iterator = vocab_entries.find(token);
      if (current_iterator == vocab_entries.end()) {
        current = new vocab_entry(0, 1);
        vocab_entries[token] = current;
        no_of_tokens++;
      } else {
        current = current_iterator->second;
        current->occurrences++;
      }
      tokens.push_back(current);
      pos = found + 1;
    }
    if (lines_read % 50000 == 0) {
      cout << "Line " << lines_read << " Complete: " << lines_read / 74004228.0
           << endl;
    }
  }
  train_data_file.close();
  cout << "Reading phase finished, number of tokens read: " << tokens.size()
       << ", vocabulary size: " << vocab_entries.size() << endl;
}

// 2. threshold the tokens based on frequency and assign uint16 values

uint32_t get_threshold(unordered_map<string, vocab_entry *> &vocab_entries,
                       size_t tokens_count) {
  if (vocab_entries.size() <= THRESHOLD) {
    size_t threshold = vocab_entries.size();
    token id = 1; // 0 token will signify unknown tokens
    for (auto &kvp : vocab_entries) {
      kvp.second->id = id;
      ++id;
    }
    return threshold;
    cout << "No thresholding was done due to the low vocabulary count "
            "(THRESHOLD = "
         << THRESHOLD << ")" << endl;

  } else {
    vector<vocab_entry *> indices;
    indices.reserve(vocab_entries.size());
    for (auto &kvp : vocab_entries) {
      indices.push_back(kvp.second);
    }

    sort(indices.begin(), indices.end(), [](vocab_entry *a, vocab_entry *b) {
      return a->occurrences > b->occurrences;
    });

    token id = 1;
    size_t covered_tokens_count = 0;
    for (size_t i = 0; i < THRESHOLD; ++i) {
      indices[i]->id = id;
      covered_tokens_count += indices[i]->occurrences;
      ++id;
    }
    // default id is 0, so the rest of them wont't have to be set

    cout << "Thresholding finished, discarded " << indices.size() - THRESHOLD
         << " vocabulary entries covering " << std::fixed
         << std::setprecision(2)
         << (covered_tokens_count * 100.0) / tokens_count
         << "% of the tokens. THRESHOLD = " << THRESHOLD << endl;

    return THRESHOLD;
  }
}

// 3. write the data to file
void write_tokens(unordered_map<string, vocab_entry *> &vocab_entries,
                  vector<vocab_entry *> &tokens,
                  string tokens_fname) {
  vector<token> tokens_buffer(4096*1024);

  ofstream tokensfile(tokens_fname, std::ios::binary);

  size_t tokens_i = 0;
  size_t num_of_tokens = tokens.size();

  tokensfile.write(reinterpret_cast<char *>(&num_of_tokens),
                   sizeof(num_of_tokens));
  while (tokens_i < tokens.size()) {
    size_t buffer_i;
    for (buffer_i = 0;
         buffer_i < tokens_buffer.size() && tokens_i < tokens.size();
         ++buffer_i, ++tokens_i) {
      tokens_buffer[buffer_i] = tokens[tokens_i]->id;
    }
    tokensfile.write(reinterpret_cast<char *>(tokens_buffer.data()),
                     buffer_i * sizeof(token));
  }

  tokensfile.close();

  cout << "Finished writing tokens to file." << endl;
}

void write_vocab(unordered_map<string, vocab_entry *> &vocab_entries,
                 string vocab_fname, uint32_t threshold) {
  ofstream vocab_file(vocab_fname);

  size_t lines_written = 0;
  for (auto &kvp : vocab_entries) {
    if (kvp.second->id == 0) {
      continue;
    }

    ++lines_written;
    vocab_file << kvp.first << ":" << kvp.second->id << endl;
    if (lines_written % 50000 == 0) {
      cout << "Vocabs written: " << lines_written
           << " Complete: " << lines_written / ((double)threshold)
           << endl;
    }
  }

  vocab_file.close();
  cout << "Finished writing vocabulary to file" << endl;
}

int main() {
  unordered_map<string, vocab_entry *> vocab_entries;
  vector<vocab_entry *> str_tokens;
  str_tokens.reserve(
      1000000000); // there is around 1 billion tokens in the corpus

  string train_data_file("data/train.txt");
  string tokens_file("data/tokens.bin");
  string vocab_file("data/vocab.txt");

  read_tokens(vocab_entries, str_tokens, train_data_file);
  auto threshold= get_threshold(vocab_entries, str_tokens.size());
  write_tokens(vocab_entries, str_tokens, tokens_file);
  write_vocab(vocab_entries, vocab_file, threshold);

  for (auto &kvp : vocab_entries) {
    delete kvp.second;
  }
}