# MissingLink.AI TensorFlow Model Zoo

In this folder you can find TensorFlow models integrated with the MissingLink.AI SDK.
The models are taken from the [TensorFlow model zoo](https://github.com/tensorflow/models) and from the [Google Cloud Platform cloudml-samples](https://github.com/GoogleCloudPlatform/cloudml-samples).

Every model comes with:
* An original `README` file, that contains explanation about the model, installation instructions and usage instructions;
* A `requirements.txt` file, that can be used with pip to install the required packages to run the model (see usage section for details).

## Usage

* Clone the repository.
* Choose the model you want to run.
* Install the required packages to run the model:
```
# from the 'model-zoo/tensorflow/<model-name>' directory
pip install -r requirements.txt
```
* Log in to the MissingLink.AI website and create a new project.
* In the file of the integrated model, find the following line:
```python
project = ml.TensorFlowProject(owner_id="your-owner-id", project_token="your-project-token")
```
* Find your owner ID and project token:

![missinglink](https://user-images.githubusercontent.com/6713560/35779613-c6fb0044-09d8-11e8-9064-38051e2157f4.png)
![missinglink](https://user-images.githubusercontent.com/6713560/35779612-c4efc244-09d8-11e8-8e12-0a516157af23.png)
* Insert your owner ID and the project token in the appropriate places.
* Run the model:
```
# from the 'model-zoo/tensorflow/<model-name>' directory
python <model's-file>.py
```

## Main Integration Files

A list of all the integrated files.

* [audioset](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/audioset)
  - vggish_train_demo.py
* [autoencoder](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/autoencoder)
  - AdditiveGaussianNoiseAutoencoderRunner.py
  - MaskingNoiseAutoencoderRunner.py
  - AutoencoderRunner.py
  - VariationalAutoencoderRunner.py
* [census](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/census)
  - tensorflowcore/trainer/task.py
* [compression-entropy_coder](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/compression-entropy_coder)
  - core/entropy_coder_train.py
* [inception](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/inception)
  - inception/inception_train.py
* [learning_to_remember_rare_events](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/learning_to_remember_rare_events)
  - train.py
* [lfads](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/lfads)
  - run_lfads.py
* [mnist](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/mnist)
  - convolutional.py
* [pcl_rl](https://github.com/missinglinkai/model-zoo/tree/master/tensorflow/pcl_rl)
  - trainer.py
