
docker build -t horovod:latest .

#docker run -it horovod:latest
docker run --mount type=bind,source=.,target=/mytmp -it horovod:latest

#
# inside the docker
#

pip install missinglink

python /mytmp/pytorch_mnist_ml.py
