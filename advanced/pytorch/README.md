# MissingLink.AI PyTorch Model Zoo

In this folder you can find PyTorch examples integrated with the MissingLink.AI SDK.
The examples are taken from the following sources:
* `pytorch-examples` - [pytorch/examples](https://github.com/pytorch/examples)
* `OpenNMT-py` - [OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)
* `kuangliu/pytorch-cifar` - [pytorch-cifar](https://github.com/kuangliu/pytorch-cifar)
* `kuangliu/pytorch-agender` - [pytorch-agender](https://github.com/kuangliu/pytorch-agender)
* `lstm-sentence-classifier` - [lstm_sentence_classifier](https://github.com/yuchenlin/lstm_sentence_classifier)
* `char-rnn-pytorch` - [char-rnn.pytorch](https://github.com/spro/char-rnn.pytorch)

Every example comes with an original `README` file, that contains explenation about the model, installation instructions and usage instructions.

## Usage

* Clone the repository.
* Choose the example you want to run.
* [Install PyTorch](http://pytorch.org/)
* Install the MissingLink.AI SDK:
```
pip install missinglink-sdk
```
**Note** that some example might require additional packages to run properly.
* Log in to the MissingLink.AI website and create a new project.
* In the training file of the example, find the following lines:
```python
OWNER_ID = 'your_owner_id'
PROJECT_TOKEN = 'your_project_token'
```
* Find your owner ID and project token:

![missinglink](https://user-images.githubusercontent.com/6713560/35779613-c6fb0044-09d8-11e8-9064-38051e2157f4.png)
![missinglink](https://user-images.githubusercontent.com/6713560/35779612-c4efc244-09d8-11e8-8e12-0a516157af23.png)
* Insert your owner ID and the project token in the appropriate places.
* Run the model:
```
# from the 'model-zoo/pytorch/<example-name>' directory
python <train-file>.py
```

## Main Integration Files

A list of all the integrated files.

- pytorch-examples
  - [dcgan](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/dcgan)
    - pytorch/pytorch-examples/dcgan/main.py
  - [fast_neural_style](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/fast_neural_style)
    - pytorch/pytorch-examples/fast_neural_style/neural_style/neural_style.py
  - [imagenet](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/imagenet)
    - pytorch/pytorch-examples/imagenet/main.py
  - [mnist](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/mnist)
    - pytorch/pytorch-examples/mnist/main.py
  - [regression](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/regression)
    - pytorch/pytorch-examples/regression/main.py
  - [reinforcement_learning](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/reinforcement_learning)
    - pytorch/pytorch-examples/reinforcement_learning/actor_critic.py
    - pytorch/pytorch-examples/reinforcement_learning/reinforce.py
  - [snli](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/snli)
    - pytorch/pytorch-examples/snli/train.py
  - [super_resolution](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/super_resolution)
    - pytorch/pytorch-examples/super_resolution/main.py
  - [vae](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/vae)
    - pytorch/pytorch-examples/vae/main.py
  - [word_language_model](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/pytorch-examples/word_language_model)
    - pytorch/pytorch-examples/word_language_model/main.py
- [OpenNMT-py](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/OpenNMT-py)
  - pytorch/OpenNMT-py/train.py
- kuangliu
  - [pytorch-cifar](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/kuangliu/pytorch-cifar)
    - pytorch/kuangliu/pytorch-agender/train.py
  - [pytorch-agender](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/kuangliu/pytorch-agender)
    - pytorch/kuangliu/pytorch-cifar/main.py
- [lstm_sentence_classifier](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/lstm_sentence_classifier)
  - pytorch/lstm_sentence_classifier/LSTM_sentence_classifier.py
  - pytorch/lstm_sentence_classifier/LSTM_sentence_classifier_minibatch.py
- [char-rnn-pytorch](https://github.com/missinglinkai/model-zoo/tree/master/pytorch/char-rnn-pytorch)
  - pytorch/char-rnn-pytorch/train.py

