# MissingLink.AI pyCaffe Model Zoo
In this folder you can find pyCaffe models integrated with the MissingLink.AI SDK. 

The models are taken from the following sources:

## Sources

* [Age Gender Deep Learning](https://github.com/GilLevi/AgeGenderDeepLearning)
* [All-Convolutional Net](https://github.com/mateuszbuda/ALL-CNN)
* [BVLC AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
* [BVLC CaffeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
* [BvLC GoogLeNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet)
* CaffeNet On Flickr Style: [source](https://github.com/BVLC/caffe/tree/master/models/finetune_flickr_style), [instructions](https://github.com/BVLC/caffe/tree/master/examples/finetune_flickr_style)
* [CaffeNet Fintuned on Oxford Flowers](https://gist.github.com/jimgoo/0179e52305ca768a601f)
* [Data Augumentation Testing](https://github.com/gombru/dataAugmentationTesting)
* [Flower Power](http://jimgoo.com/flower-power/)
* [GoogLeNet Cars Dataset](https://gist.github.com/bogger/b90eb88e31cd745525ae)
* [ImageNet With Batch Normalization](https://github.com/cvjena/cnn-models)
* [Network In Network](https://gist.github.com/mavenlin/e56253735ef32c3c296d)
* [SqueezeNet](https://github.com/DeepScale/SqueezeNet)
* [Translating Video to Natural Language](https://gist.github.com/vsubhashini/3761b9ad43f60db9ac3d)
* [Yearbook Photo](https://gist.github.com/katerakelly/842f948d568d7f1f0044)

## Usage

* Clone the repository.
* Choose the model you want to run.
* Log in to the MissingLink.AI website and create a new project.
* Find the following lines in the integrated file:
```python
missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
```
* Find your owner ID and project token:

![missinglink](https://user-images.githubusercontent.com/6713560/35779613-c6fb0044-09d8-11e8-9064-38051e2157f4.png)
![missinglink](https://user-images.githubusercontent.com/6713560/35779612-c4efc244-09d8-11e8-8e12-0a516157af23.png)
* Insert your owner ID and your project token in the appropriate places.
* You will probably need to provide training and testing data, and to insert the path to the data in the net's defenition `.prototxt` file.
* Run the model:
```
# from the 'model-zoo/pyCaffe/<model-name>' directory
python <train OR solve>.py
```

**Note:** Some models include a `README` with more instructions required to run them.

## Main Integration Files

* [Age Gender Deep Learning](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/age-gender-deep-learning)
  * age-gender-deep-learning/age_net_definitions/train.py
  * age-gender-deep-learning/gender_net_definitions/train.py
* [All-Convolutional Net](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/all-convolutional-net)
  * all-convolutional-net/solve.py
* [BVLC AlexNet](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/bvlc-alexnet)
  * BVLC-AlexNet/solve.py
* [BVLC CaffeNet](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/bvlc-reference-CaffeNet)
  * BVLC-reference-CaffeNet/solve.py
* [BvLC GoogLeNet](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/bvlc-GoogLeNet)
  * BVLC-GoogLeNet/solve.py
* [Caffe Net Flickr Style](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/caffe-net-flickr-style/train.py)
  * caffe-net-flickr-style/train.py
* [CaffeNet Fintuned on Oxford Flowers](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/CaffeNet-fine-tuned-for-Oxford-flowers-dataset)
  * CaffeNet-fine-tuned-for-Oxford-flowers-dataset/solve.py
* [Data Augmentation Testing](https://github.com/missinglinkai/model-zoo/blob/feature/pycaffe/pycaffe/data-augmentation-testing/train.py)
  * data-augmentation-testing/train.py
* [Flower Power](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/flower-power)
  * flower-power/AlexNet/solve.py
  * flower-power/VGG_S/solve.py
* [GoogLeNet Cars Dataset](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/GoogLeNet-cars)
  * GoogLeNet-cars/solve.py
* [ImageNet With Batch Normalization](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/ImageNet-with-batch-normlization)
  * ImageNet-with-batch-normlization/AlexNet_cvgj/train.py
  * ImageNet-with-batch-normlization/ResNet_preact/ResNet10_cvgj/train.py
  * ImageNet-with-batch-normlization/ResNet_preact/ResNet50_cvgj/train.py
  * ImageNet-with-batch-normlization/VGG19_cvgj/train.py
* [Network In Network](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/network-in-network)
  * network-in-network/train.py
* [SqueezeNet](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/SqueezeNet)
  * SqueezeNet/SqueezeNet_v1.0/solve.py
  * SqueezeNet/SqueezeNet_v1.1/solve.py
* [Translating Video to Natural Language](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/translating-video-to-natural-language):
  * translating-video-to-natural-language/solve.py
* [Yearbook Photo](https://github.com/missinglinkai/model-zoo/tree/feature/pycaffe/pycaffe/yearbook-photo):
  * yearbook-photo/solve.py
