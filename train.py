import numpy as np
import argparse
import torch
import torch.nn as nn
from custom_dataset import get_data_loaders
from preprocess_data import get_data_frames, get_vocab_and_embed_matrix
from model import Net
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt


def train_epoch(model, data_loader, device, criterion, optimizer, scheduler, num_samples):
    model.train()

    losses = []
    correct = 0

    for batch_idx, (q1, q2, targets) in enumerate(tqdm(data_loader)):
        q1 = q1.to(device)
        q2 = q2.to(device)
        targets = targets.to(device)

        output = model(q1.long(), q2.long())
        output = output.squeeze()
        loss = criterion(output.float(), targets.float())

        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        output = output >= 0.5
        correct += (output == targets).sum().item()

    acc = (correct * 1.0) / num_samples

    scheduler.step(acc)

    return acc, np.mean(losses)


def val_epoch(model, data_loader, device, criterion, num_samples):
    model.eval()

    losses = []
    correct = 0

    with torch.no_grad():
        for batch_idx, (q1, q2, targets) in enumerate(tqdm(data_loader)):
            q1 = q1.to(device)
            q2 = q2.to(device)
            targets = targets.to(device)

            output = model(q1.long(), q2.long())
            output = output.squeeze()
            loss = criterion(output.float(), targets.float())

            losses.append(loss.item())

            output = output >= 0.5
            correct += (output == targets).sum().item()

    return (correct * 1.0) / num_samples, np.mean(losses)


def train(model, EPOCHS, device, train_loader, val_loader, criterion, optimizer, scheduler):
    history = defaultdict(list)
    best_val_acc = 0

    for epoch in range(EPOCHS):
        print(f'Epoch: {epoch + 1}/{EPOCHS}')
        print('-' * 10)
        print('Training')
        train_acc, train_loss = train_epoch(model, train_loader, device, criterion, optimizer, scheduler,
                                            len(train_loader.sampler))
        print('Validating')
        val_acc, val_loss = val_epoch(model, val_loader, device, criterion, len(val_loader.sampler))
        print(f'Train Loss: {train_loss}\tTrain Acc: {train_acc}')
        print(f'Val Loss: {val_loss}\tVal Acc: {val_acc}')

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_model.pth.tar')
    return history


def test_model(model, data_loader, device):
    model.eval()

    predictions = []

    with torch.no_grad():
        for batch_idx, (q1, q2) in enumerate(tqdm(data_loader)):
            q1 = q1.to(device)
            q2 = q2.to(device)

            output = model(q1.long(), q2.long())  # (batch_size, 1)
            output = output.squeeze()
            predictions.append(output)

    return predictions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=32)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--gpuidx', type=int, default=0)
    parser.add_argument('--train_val_split', type=float, default=0.8)
    parser.add_argument('--embed_dim', type=int, default=100)
    parser.add_argument('--sample_rows_per_class', type=int, default=10000)
    parser.add_argument('--train_model', type=int, default=1)
    parser.add_argument('--test_model', type=int, default=0)
    parser.add_argument('--pretrained_embed', type=int, default=1)

    args = parser.parse_args()

    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    NUM_WORKERS = args.num_workers
    LEARNING_RATE = args.learning_rate
    GPUIDX = args.gpuidx
    TRAIN_VAL_SPLIT = args.train_val_split
    EMBED_DIM = args.embed_dim
    SAMPLE_ROWS_PER_CLASS = args.sample_rows_per_class
    TRAIN_MODEL = args.train_model
    TEST_MODEL = args.test_model
    PRETRAINED_EMBED = args.pretrained_embed

    DEVICE = f'cuda:{GPUIDX}' if torch.cuda.is_available() else 'cpu'

    train_df, test_df, sample_sub_df = get_data_frames(sample_rows_per_class=SAMPLE_ROWS_PER_CLASS)
    vocab, embed_matrix = get_vocab_and_embed_matrix(train_df, EMBED_DIM)
    print(f'Vocab size: {len(vocab.vocab)}')
    train_loader, val_loader, test_loader = get_data_loaders(vocab,
                                                             train_df,
                                                             test_df,
                                                             BATCH_SIZE,
                                                             TRAIN_VAL_SPLIT,
                                                             NUM_WORKERS,
                                                             )
    vocab_size = len(vocab.vocab)
    model = Net(vocab_size, embed_matrix, pretrained_embed=PRETRAINED_EMBED).to(DEVICE)

    criterion = nn.BCELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, nesterov=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2, verbose=False)

    if TRAIN_MODEL:
        history = train(model, EPOCHS, DEVICE, train_loader, val_loader, criterion, optimizer, scheduler)

        plt.figure()
        plt.plot(history['train_loss'], label='train loss')
        plt.plot(history['val_loss'], label='val loss')
        plt.title('Training Loss vs Validation Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('quora_train_val_loss.png')

        plt.figure()
        plt.plot(history['train_acc'], label='train acc')
        plt.plot(history['val_acc'], label='val acc')
        plt.title('Training Acc vs Validation Acc')
        plt.ylabel('Acc')
        plt.xlabel('Epoch')
        plt.legend()
        plt.savefig('quora_train_val_acc.png')

    if TEST_MODEL:
        print('-' * 10)
        print('-' * 10)
        print('Testing')
        print('-' * 10)
        print('-' * 10)

        # Load best model
        model.load_state_dict(torch.load('best_model.pth.tar'))
        predictions = test_model(model, test_loader, DEVICE)
        preds = torch.cat(predictions)
        preds = preds.cpu().numpy()
        sample_sub_df['is_duplicate'] = preds
        sample_sub_df.to_csv('submission.csv', index=False)


if __name__ == "__main__":
    main()
