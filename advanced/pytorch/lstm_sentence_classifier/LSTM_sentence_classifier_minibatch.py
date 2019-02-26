# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchtext import data
import classification_datasets
import os
import random
torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)
# torch.cuda.set_device(args.gpu)
import torch.utils.data as Data

import missinglink

OWNER_ID = 'your_owner_id'
PROJECT_TOKEN = 'your_project_token'

missinglink_project = missinglink.PyTorchProject(owner_id=OWNER_ID, project_token=PROJECT_TOKEN)

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        return (autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, self.batch_size, self.hidden_dim)))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y)
        return log_probs

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0
     for i in range(len(truth)):
         if truth[i]==pred[i]:
             right += 1.0
     return right/len(truth)

wrapped_accuracy_function = None

def train():
    EMBEDDING_DIM = 100
    HIDDEN_DIM = 50
    EPOCH = 100
    BATCH_SIZE = 10
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)
    train_iter, dev_iter , test_iter = classification_datasets.load_mr(text_field, label_field, batch_size=BATCH_SIZE)

    # text_field.vocab.load_vectors('glove.6B.50d')
    text_field.vocab.load_vectors('glove.6B.100d')

    best_dev_acc = 0.0

    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM, hidden_dim=HIDDEN_DIM,
                           vocab_size=len(text_field.vocab),label_size=len(label_field.vocab)-1,
                            batch_size=BATCH_SIZE)
    model.word_embeddings.weight.data = text_field.vocab.vectors
    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)

    with missinglink_project.create_experiment(
        model,
        display_name='LSTM Sentence Classifier PyTorch',
        optimizer=optimizer,
        metrics={
            'Loss': loss_function,
            'Accuracy': get_accuracy,
        }
    ) as experiment:
        wrapped_loss_function = experiment.metrics['Loss']
        global wrapped_accuracy_function
        wrapped_accuracy_function = experiment.metrics['Accuracy']

        no_up = 0
        for i in experiment.epoch_loop(condition=lambda epoch: epoch < EPOCH and no_up < 10):
            print('epoch: %d start!' % i)
            train_epoch(experiment, model, train_iter, wrapped_loss_function, optimizer, text_field, label_field, i)
            print('now best dev acc:',best_dev_acc)
            dev_acc = evaluate(experiment, model,dev_iter,wrapped_loss_function,'dev')
            test_acc = evaluate(experiment, model, test_iter, wrapped_loss_function,'test')
            if dev_acc > best_dev_acc:
                best_dev_acc = dev_acc
                os.system('rm best_models/mr_best_model_minibatch_acc_*.model')
                print('New Best Dev!!!')
                torch.save(model.state_dict(), 'best_models/mr_best_model_minibatch_acc_' + str(int(test_acc*10000)) + '.model')
                no_up = 0
            else:
                no_up += 1

def evaluate(experiment, model, eval_iter, loss_function,  name ='dev'):
    model.eval()
    avg_loss = 0.0
    truth_res = []
    pred_res = []

    if name == 'dev':
        context = experiment.validation()
    else:
        context = experiment.test(test_iterations=len(eval_iter))

    with context:
        for batch in eval_iter:
            sent, label = batch.text, batch.label
            label.data.sub_(1)
            truth_res += list(label.data)
            model.batch_size = len(label.data)
            model.hidden = model.init_hidden()  # detaching it from its history on the last instance.
            pred = model(sent)
            pred_label = pred.data.max(1)[1].numpy()
            # pred_res += [x[0] for x in pred_label]
            pred_res += [x for x in pred_label]
            loss = loss_function(pred, label)
            avg_loss += loss.data[0]

            if name != 'dev':
                experiment.confusion_matrix(target=label, output=pred)

        avg_loss /= len(eval_iter)
        global wrapped_accuracy_function
        acc = wrapped_accuracy_function(truth_res, pred_res)
        print(name + ' avg_loss:%g train acc:%g' % (avg_loss, acc ))
        return acc

def train_epoch(experiment, model, train_iter, loss_function, optimizer, text_field, label_field, i):
    model.train()
    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    for _, batch in experiment.batch_loop(iterable=train_iter):
        sent, label = batch.text, batch.label
        label.data.sub_(1)
        truth_res += list(label.data)
        model.batch_size = len(label.data)
        model.hidden = model.init_hidden()# detaching it from its history on the last instance.
        pred = model(sent)
        pred_label = pred.data.max(1)[1].numpy()
        # pred_res += [x[0] for x in pred_label]
        pred_res += [x for x in pred_label]
        model.zero_grad()
        loss = loss_function(pred, label)
        avg_loss += loss.data[0]
        count += 1
        if count % 100 == 0:
            print('epoch: %d iterations: %d loss :%g' % (i, count*model.batch_size, loss.data[0]))
        loss.backward()
        optimizer.step()
    avg_loss /= len(train_iter)
    global wrapped_accuracy_function
    print('epoch: %d done!\ntrain avg_loss:%g , acc:%g'%(i, avg_loss, wrapped_accuracy_function(truth_res,pred_res)))

train()
