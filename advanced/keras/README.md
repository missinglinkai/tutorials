# MissingLink.AI Keras Model Zoo

In this folder you can find Keras models integrated with the MissingLink.AI SDK. The models are taken from the [Keras examples directory](https://github.com/fchollet/keras/tree/master/examples).
This folder includes:
* The integrated models
* This `README` file
* A `requierments.txt` file

## Usage

* Clone the repository.
* Install the requirements to run the models:
```
# from the 'model-zoo/keras' directory:
pip install -r requirements.txt
```
* Choose the model you want to run.
* Log in to the MissingLink.AI website and create a new project.
* Find the following lines in the model's file:
```python
missinglink_callback = missinglink.KerasCallback(
    owner_id="your-owner-id",
    project_token="your-project-token"
)
```
* Find your owner ID and project token:

![missinglink](https://user-images.githubusercontent.com/6713560/35779613-c6fb0044-09d8-11e8-9064-38051e2157f4.png)
![missinglink](https://user-images.githubusercontent.com/6713560/35779612-c4efc244-09d8-11e8-8e12-0a516157af23.png)
* Insert your owner ID and your project token in the appropriate places.
* Run the model:
```
# from the 'model-zoo/keras' directory:
python <model-name>.py
```

## Models

[addition_rnn.py](addition_rnn.py)
Implementation of sequence to sequence learning for performing addition of two numbers (as strings).

[antirectifier.py](antirectifier.py)
Demonstrates how to write custom layers for Keras.

[babi_memnn.py](babi_memnn.py)
Trains a memory network on the bAbI dataset for reading comprehension.

[babi_rnn.py](babi_rnn.py)
Trains a two-branch recurrent network on the bAbI dataset for reading comprehension.

[cifar10_cnn.py](cifar10_cnn.py)
Trains a simple deep CNN on the CIFAR10 small images dataset.

[conv_lstm.py](conv_lstm.py)
Demonstrates the use of a convolutional LSTM network.

[image_ocr.py](image_ocr.py)
Trains a convolutional stack followed by a recurrent stack and a CTC logloss function to perform optical character recognition (OCR).

[imdb_bidirectional_lstm.py](imdb_bidirectional_lstm.py)
Trains a Bidirectional LSTM on the IMDB sentiment classification task.

[imdb_cnn.py](imdb_cnn.py)
Demonstrates the use of Convolution1D for text classification.

[imdb_cnn_lstm.py](imdb_cnn_lstm.py)
Trains a convolutional stack followed by a recurrent stack network on the IMDB sentiment classification task.

[imdb_fasttext.py](imdb_fasttext.py)
Trains a FastText model on the IMDB sentiment classification task.

[imdb_lstm.py](imdb_lstm.py)
Trains an LSTM model on the IMDB sentiment classification task.

[lstm_benchmark.py](lstm_benchmark.py)
Compares different LSTM implementations on the IMDB sentiment classification task.

[lstm_text_generation.py](lstm_text_generation.py)
Generates text from Nietzsche's writings.

[mnist_cnn.py](mnist_cnn.py)
Trains a simple convnet on the MNIST dataset.

[mnist_hierarchical_rnn.py](mnist_hierarchical_rnn.py)
Trains a Hierarchical RNN (HRNN) to classify MNIST digits.

[mnist_irnn.py](mnist_irnn.py)
Reproduction of the IRNN experiment with pixel-by-pixel sequential MNIST in "A Simple Way to Initialize Recurrent Networks of Rectified Linear Units" by Le et al.

[mnist_mlp.py](mnist_mlp.py)
Trains a simple deep multi-layer perceptron on the MNIST dataset.

[mnist_siamese_graph.py](mnist_siamese_graph.py)
Trains a Siamese multi-layer perceptron on pairs of digits from the MNIST dataset.

[mnist_swwae.py](mnist_swwae.py)
Trains a Stacked What-Where AutoEncoder built on residual blocks on the MNIST dataset.

[mnist_tfrecord.py](mnist_tfrecord.py)
MNIST dataset with TFRecords, the standard TensorFlow data format.

[mnist_transfer_cnn.py](mnist_transfer_cnn.py)
Transfer learning toy example.

[reuters_mlp.py](reuters_mlp.py)
Trains and evaluate a simple MLP on the Reuters newswire topic classification task.

[reuters_mlp_relu_vs_selu.py](reuters_mlp_relu_vs_selu.py)
Compares self-normalizing MLPs with regular MLPs.

[stateful_lstm.py](stateful_lstm.py)
Demonstrates how to use stateful RNNs to model long sequences efficiently.

[variational_autoencoder.py](variational_autoencoder.py)
Demonstrates how to build a variational autoencoder.

[variational_autoencoder_deconv.py](variational_autoencoder_deconv.py)
Demonstrates how to build a variational autoencoder with Keras using deconvolution layers.
