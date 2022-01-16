import torch
import torch.nn as nn


def create_embed_layer(embed_matrix, non_trainable=False):
    num_embeddings, embedding_dim = embed_matrix.shape
    embed_layer = nn.Embedding(num_embeddings, embedding_dim)
    embed_layer.load_state_dict({'weight': embed_matrix})
    if non_trainable:
        embed_layer.weight.requires_grad = False

    return embed_layer, num_embeddings, embedding_dim


class Net(nn.Module):
    def __init__(self, vocab_size, embed_matrix, num_layers=2, embed_dim=200, hidden_size=256, pretrained_embed=True,
                 bidirectional=False):
        super(Net, self).__init__()
        self.hidden_size = hidden_size
        if pretrained_embed:
            self.embedding, num_embeddings, embed_dim = create_embed_layer(embed_matrix)
        else:
            self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_dim)

        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(embed_dim, hidden_size,
                            num_layers=num_layers,
                            batch_first=False,
                            bidirectional=self.bidirectional)

        if self.bidirectional:
            self.fc1 = nn.Linear(hidden_size * 4, 128)
        else:
            self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, q1, q2):
        # shape of x: (seq_len, batch_size)
        seq_len, batch_size = q1.shape
        q1_embeddings = self.embedding(q1)
        q2_embeddings = self.embedding(q2)
        _, (q1_hidden, _) = self.lstm(q1_embeddings)
        _, (q2_hidden, _) = self.lstm(q2_embeddings)
        # shape of hidden, cell: (num_layers * num_directions, batch_size, hidden_size)

        if self.bidirectional:
            q1_hidden = q1_hidden.reshape(self.num_layers, 2, batch_size, self.hidden_size)
            # shape of q1_hidden: (num_layers, num_directions, batch_size, hidden_size)
            q2_hidden = q2_hidden.reshape(self.num_layers, 2, batch_size, self.hidden_size)
            # shape of q2_hidden: (num_layers, num_directions, batch_size, hidden_size)
            q1_hidden = q1_hidden[-1]  # get the last layer hidden state
            q2_hidden = q2_hidden[-1]  # get the last layer hidden state
            # shape of hidden: (num_directions, batch_size, hidden_size)
            hidden = torch.cat([q1_hidden, q2_hidden], dim=2)
            hidden = hidden.reshape(hidden.shape[1], -1)  # (batch_size, num_directions * hidden_size)
        else:
            hidden = q1_hidden[-1]  # (batch_size, hidden_size)

        out = self.dropout(self.relu(self.fc1(hidden)))  # (batch_size, 128)
        out = self.fc2(out)  # (batch_size, 1)
        return self.sigmoid(out)

def main():
    q1 = torch.zeros((35, 32)).long()
    q2 = torch.zeros((35, 32)).long()
    targets = torch.zeros(32)
    vocab_size = 10
    embed_matrix = torch.zeros((10, 200))

    model = Net(vocab_size, embed_matrix, bidirectional=False)
    out = model(q1,q2)
    out = out.squeeze()

    criterion = nn.BCELoss()
    loss = criterion(out.float(), targets.float())
    print(loss)


if __name__ == "__main__":
    main()