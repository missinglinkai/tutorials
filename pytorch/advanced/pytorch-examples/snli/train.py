import os
import time
import glob

import torch
import torch.optim as O
import torch.nn as nn

from torchtext import data
from torchtext import datasets

from model import SNLIClassifier
from util import get_args, makedirs

import missinglink

OWNER_ID = 'your_owner_id'
PROJECT_TOKEN = 'your_project_token'

missinglink_project = missinglink.PyTorchProject(owner_id=OWNER_ID, project_token=PROJECT_TOKEN)


args = get_args()
torch.cuda.set_device(args.gpu)

inputs = data.Field(lower=args.lower)
answers = data.Field(sequential=False)

print('Loading data')
train, dev, test = datasets.SNLI.splits(inputs, answers)

inputs.build_vocab(train, dev, test)
if args.word_vectors:
    if os.path.isfile(args.vector_cache):
        inputs.vocab.vectors = torch.load(args.vector_cache)
    else:
        inputs.vocab.load_vectors(args.word_vectors)
        makedirs(os.path.dirname(args.vector_cache))
        torch.save(inputs.vocab.vectors, args.vector_cache)
answers.build_vocab(train)

train_iter, dev_iter, test_iter = data.BucketIterator.splits(
            (train, dev, test), batch_size=args.batch_size, device=args.gpu)

config = args
config.n_embed = len(inputs.vocab)
config.d_out = len(answers.vocab)
config.n_cells = config.n_layers

# double the number of cells for bidirectional networks
if config.birnn:
    config.n_cells *= 2

print('Loading model')
if args.resume_snapshot:
    model = torch.load(args.resume_snapshot, map_location=lambda storage, locatoin: storage.cuda(args.gpu))
else:
    model = SNLIClassifier(config)
    if args.word_vectors:
        model.embed.weight.data = inputs.vocab.vectors
        # model.cuda()

criterion = nn.CrossEntropyLoss()
opt = O.Adam(model.parameters(), lr=args.lr)

iterations = 0
start = time.time()
best_dev_acc = -1
train_iter.repeat = False
header = '  Time Epoch Iteration Progress    (%Epoch)   Loss   Dev/Loss     Accuracy  Dev/Accuracy'
dev_log_template = ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{:8.6f},{:12.4f},{:12.4f}'.split(','))
log_template =     ' '.join('{:>6.0f},{:>5.0f},{:>9.0f},{:>5.0f}/{:<5.0f} {:>7.0f}%,{:>8.6f},{},{:12.4f},{}'.split(','))
makedirs(args.save_path)
print(header)

with missinglink_project.create_experiment(
    model,
    display_name='SNLI PyTorch',
    optimizer=opt,
    metrics={
        'Loss': criterion,
        'Accuracy': lambda correct, total: 100. * correct / total
    }
) as experiment:
    wrapped_loss = experiment.metrics['Loss']
    wrapped_accuracy = experiment.metrics['Accuracy']

    for epoch in experiment.epoch_loop(args.epochs):
        train_iter.init_epoch()
        n_correct, n_total = 0, 0
        for batch_idx, batch in experiment.batch_loop(iterable=train_iter):

            # switch model to training mode, clear gradient accumulators
            model.train(); opt.zero_grad()

            iterations += 1

            # forward pass
            answer = model(batch)

            # calculate accuracy of predictions in the current batch
            n_correct += (torch.max(answer, 1)[1].view(batch.label.size()).data == batch.label.data).sum()
            n_total += batch.batch_size
            train_acc = wrapped_accuracy(n_correct, n_total)

            # calculate loss of the network output with respect to training labels
            loss = wrapped_loss(answer, batch.label)

            # backpropagate and update optimizer learning rate
            loss.backward(); opt.step()

            # checkpoint model periodically
            if iterations % args.save_every == 0:
                snapshot_prefix = os.path.join(args.save_path, 'snapshot')
                snapshot_path = snapshot_prefix + '_acc_{:.4f}_loss_{:.6f}_iter_{}_model.pt'.format(train_acc, loss.data[0], iterations)
                torch.save(model, snapshot_path)
                for f in glob.glob(snapshot_prefix + '*'):
                    if f != snapshot_path:
                        os.remove(f)

            # evaluate performance on validation set periodically
            if iterations % args.dev_every == 0:
                with experiment.validation():
                    # switch model to evaluation mode
                    model.eval(); dev_iter.init_epoch()

                    # calculate accuracy on validation set
                    n_dev_correct, dev_loss = 0, 0
                    for dev_batch_idx, dev_batch in enumerate(dev_iter):
                         answer = model(dev_batch)
                         n_dev_correct += (torch.max(answer, 1)[1].view(dev_batch.label.size()).data == dev_batch.label.data).sum()
                         dev_loss = wrapped_loss(answer, dev_batch.label)
                    dev_acc = wrapped_accuracy(n_dev_correct, len(dev))

                    print(dev_log_template.format(time.time()-start,
                        epoch, iterations, 1+batch_idx, len(train_iter),
                        100. * (1+batch_idx) / len(train_iter), loss.data[0], dev_loss.data[0], train_acc, dev_acc))

                    # update best valiation set accuracy
                    if dev_acc > best_dev_acc:

                        # found a model with better validation set accuracy

                        best_dev_acc = dev_acc
                        snapshot_prefix = os.path.join(args.save_path, 'best_snapshot')
                        snapshot_path = snapshot_prefix + '_devacc_{}_devloss_{}__iter_{}_model.pt'.format(dev_acc, dev_loss.data[0], iterations)

                        # save model, delete previous 'best_snapshot' files
                        torch.save(model, snapshot_path)
                        for f in glob.glob(snapshot_prefix + '*'):
                            if f != snapshot_path:
                                os.remove(f)

            elif iterations % args.log_every == 0:

                # print progress message
                print(log_template.format(time.time()-start,
                    epoch, iterations, 1+batch_idx, len(train_iter),
                    100. * (1+batch_idx) / len(train_iter), loss.data[0], ' '*8, n_correct/n_total*100, ' '*12))

        with experiment.test(test_data_object=test_iter, target_attribute_name='label'):
            # switch model to evaluation mode
            model.eval()
            dev_iter.init_epoch()

            # calculate accuracy on test set
            n_test_correct, test_loss = 0, 0
            for test_batch_idx, test_batch in enumerate(test_iter):
                answer = model(test_batch)
                n_test_correct += (torch.max(answer, 1)[1].view(test_batch.label.size()).data == test_batch.label.data).sum()
                test_loss += wrapped_loss(answer, test_batch.label)
            test_acc = wrapped_accuracy(n_test_correct, len(test_iter))
            test_loss /= len(test_iter)

            print('Test: Epoch %d  Loss %f  Accuracy %f' % (epoch, test_loss[0], test_acc))
