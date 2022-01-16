import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from spacy.lang.en import English
nlp = English()
tokenizer = nlp.tokenizer


def tokenize(text):
    tokens = tokenizer(text)
    return tokens

class CustomDataset(Dataset):
    def __init__(self, vocab, df, train=True):
        self.vocab = vocab
        self.df = df
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        if self.train:
            q1 = self.df.iloc[index, 3]
            q2 = self.df.iloc[index, 4]
        else:
            q1 = self.df.iloc[index, 1]
            q2 = self.df.iloc[index, 2]

        q1_tokens = tokenize(q1.lower())
        q2_tokens = tokenize(q2.lower())

        q1_encoded = self.encode(q1_tokens)
        q2_encoded = self.encode(q2_tokens)

        if self.train:
            target = self.df.iloc[index, 5]
            if int(target) == -1:
                target = 0
            return torch.tensor(q1_encoded).long(), torch.tensor(q2_encoded).long(), torch.tensor(target).long()
        else:
            return torch.tensor(q1_encoded).long(), torch.tensor(q2_encoded).long()

    def encode(self, tokens):
        encoded = [self.vocab.stoi[str(tok)] if str(tok) in self.vocab.stoi else self.vocab.stoi['<UNK>'] for tok in
                   tokens]
        return encoded


class MyCollate:
    def __init__(self, pad_idx, train=True):
        self.pad_idx = pad_idx
        self.train = train

    def __call__(self, batch):
        q1 = [item[0] for item in batch]
        q2 = [item[1] for item in batch]
        q1 = nn.utils.rnn.pad_sequence(q1, batch_first=False, padding_value=self.pad_idx)
        q2 = nn.utils.rnn.pad_sequence(q2, batch_first=False, padding_value=self.pad_idx)
        if self.train:
            targets = [item[2] for item in batch]
            return q1, q2, torch.tensor(targets)
        else:
            return q1, q2


def get_data_loaders(vocab, train_df, test_df, batch_size=32, train_val_split=0.8, num_workers=0):
    train_dataset = CustomDataset(vocab, train_df)
    test_dataset = CustomDataset(vocab, test_df, train=False)
    train_size = int(train_val_split * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    pad_idx = vocab.stoi['<PAD>']
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              collate_fn=MyCollate(pad_idx, train=True),
                              num_workers=num_workers
                              )

    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            collate_fn=MyCollate(pad_idx, train=True),
                            num_workers=num_workers
                            )

    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=batch_size,
                             collate_fn=MyCollate(pad_idx, train=False)
                             )

    return train_loader, val_loader, test_loader