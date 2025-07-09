import torch
import torch.nn as nn
import torch.nn.functional as F
    
class BooksModel(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super(BooksModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h):
        x = self.embed(x)
        out, h = self.rnn(x, h)
        out = self.fc(out)
        return out, h

    def predict(self, x):
        with torch.no_grad():
            self.eval()
            out, h = self.forward(x)
            _, predicted = torch.max(out, 1)
        return predicted

    def name(self):
        return self.__class__.__name__