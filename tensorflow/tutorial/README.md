# Introduction

In this tutorial we will take the existing implementation of a deep learning algorithm and integrate it into the MissingLink system. 

We start with a [code sample](https://github.com/tensorflow/models/tree/master/tutorials/image/mnist) that trains a model based on the MNIST dataset using a convolutional neural network, add the MissingLink SDK, and eventually run the experiment in a MissingLink controlled emulated server.

# Getting Started

## Prerequisites

To run this tutorial, you will need a MissingLink account. If you don't have one, [head to the MissingLink website and sign up](https://missinglink.ai/console/signup/userdetails).

You will also need to have [Python](https://www.python.org/downloads/) and [Docker](https://docs.docker.com/install/#supported-platforms) installed on your workstation.

---
**NOTE**

This tutorial assumes you’re using virtualenv to scope your working environment.
If you don't have it installed, you can follow [this guide](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/) to get it set up.

---

## First things first ...

Let’s head to the MissingLink TensorFlow Tutorial 1 [Github repository](https://github.com/missinglinkai/missinglink-tensorflow-tutorial1) and examine it.

 Notice it contains the program file, `mnist_cnn.py`, and a `requirements.txt` file. This code trains a simple convnet on the MNIST dataset (borrowed from [TensorFlow tutorials](https://github.com/tensorflow/models/tree/master/tutorials/image/mnist)).  

To make changes, you will need to create a copy of the repo and fetch it to your local development environment. Go ahead and create a fork of the [tutorial repository](https://github.com/missinglinkai/missinglink-tensorflow-tutorial1) by clicking Fork.

![Fork on Github](./images/fork_repo.png)

After the forked repository is created, clone it locally in your workstation. Click Clone or download in Github:

![Fork on Github](./images/clone_button.png)

Now copy the URL for cloning the repository:

![Copy repo url](./images/copy_repo_url_button.png)

Next, let’s open a terminal and `git clone` using the pasted URL of your forked repository:  

```bash
$ git clone git@github.com:<YOUR_GITHUB_USERNAME>/missinglink-tensorflow-tutorial1.git
$ cd missinglink-tensorflow-tutorial1
```

Now that the code is on your machine, let's prepare the environment. Run the following commands:

```bash
$ python3 -m virtualenv env
$ source env/bin/activate
$ pip install -r requirements.txt
```

## Let's run it

You can try to run the example:

```bash
$ python mnist_cnn.py
```

![Experiment progress in terminal](./images/tutorials-experiment-start.gif)

As you can see, the code starts running the experiment.

# Integrating the MissingLink SDK

Now, let's see how, by adding a few lines of code and a few commands, we're able to follow the experiment in MissingLink's web dashboard.

## Install and initialize the MissingLink CLI

MissingLink provides a command line interface (CLI) that allows you to control everything from the terminal.

Let's go ahead and install it:

```bash
$ pip install missinglink
```

Next, authenticate with the MissingLink backend.

---
**NOTE**

Once you run the following command, a browser window launches and accesses the MissingLink website.

If you're not logged on, you will be asked to log on. When the process is completed, you will get a message to go back to the terminal.


---

```bash
$ ml auth init
```

## Creating a project

MissingLink allows you to manage several projects. Let's create a new project for this tutorial:

```bash
$ ml projects create --display-name tutorials
```

---
**NOTE**

You can see a list of all your projects by running `ml projects list`, or obviously, by going to the [MissingLink web dashboard](https://missinglink.ai/console).

---

## Updating the requirements

Open the code in your favorite IDE.
Add the MissingLink SDK as a requirement in the `requirements.txt` file:

```diff
tensorflow
+missinglink
```

## Create the experiment in MissingLink

Open the `mnist_cnn.py` script file and import the MissingLink SDK:
```diff
// ...
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
+import missinglink

# Input params
NUM_CLASSES = 10  # The MNIST dataset has 10 classes, representing the digits 0 through 9.
IMAGE_SIZE = 28  # The MNIST images are always 28x28 pixels.
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
// ...
```

Now we need to initialize the project so that we can have TensorFlow call the MissingLink server during the different stages of the experiment.

```diff
// ...
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import missinglink
+
+project = missinglink.TensorFlowProject()

# Input params
NUM_CLASSES = 10  # The MNIST dataset has 10 classes, representing the digits 0 through 9.
IMAGE_SIZE = 28  # The MNIST images are always 28x28 pixels.
IMAGE_PIXELS = IMAGE_SIZE * IMAGE_SIZE
// ...
```

Now let's have TensorFlow use our project. We would need to inject calls to MissingLink during the training stage.  
Let's scroll to the `run_training()` function and replace the step loop with a MissingLink experiment loop:

```diff
// ...
        # Initialize the graph
        init = tf.global_variables_initializer()
        session = tf.Session()
        session.run(init)

+
+       with project.create_experiment() as experiment:
            # Start the training loop
-           for step in range(MAX_STEPS):
+           for step in experiment.loop(max_iterations=MAX_STEPS):
                feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

                _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)
// ...
```

Next, we need to wrap the training call with a call to the SDK. We'll `experiment.train` scope before the `session.run` which runs the optimizer to let the SDK know it should collect the metrics as training metrics.

```diff
// ...
            for step in experiment.loop(max_iterations=MAX_STEPS):
                feed_dict = fill_feed_dict(data_sets.train, images_placeholder, labels_placeholder)

-               _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)
+               with experiment.train(monitored_metrics={'loss': loss, 'acc': eval_correct}):
+                   _, loss_value = session.run([train_op, loss], feed_dict=feed_dict)

                # Validate the model with the validation dataset
                if (step + 1) % 500 == 0 or (step + 1) == MAX_STEPS:
// ...
```

Lastly, let the MissingLink SDK know we're running validation:

```diff
// ...
                # Validate the model with the validation dataset
                if (step + 1) % 500 == 0 or (step + 1) == MAX_STEPS:
                    print('Step %d: loss = %.2f' % (step, loss_value))
                    print('Running on validation dataset...')
+                   with experiment.validation(monitored_metrics={'loss': loss, 'acc': eval_correct}):
                        do_eval(session, eval_correct, images_placeholder, labels_placeholder, data_sets.validation)

// ...
```

## Run the integrated experiment
We're all set up to run the experiment again, but this time to see it in the Missing Link dashboard.  

Go back to the terminal and run the script again:

```bash
$ python mnist_cnn.py
```

You should see the initialization and the beginning of training. Now, switch back to the MissingLink dashboard.

Open the [MissingLink dashboard](https://missinglink.ai/console) and click the projects toolbar button on the left. In this page, you should see the list of experiments that belong to your project.

![List of projects](./images/project_list_tutorials_project.png)

Choose the **tutorials** project. Your experiment appears.  

![Experiment in list](./images/tutorial_experiment.png)

Now you can click anywhere on the experiment line to show more information about the experiment's progress.

![Experiment detailed view](./images/tutorials_experiment_info.png)

---
**NOTE**

Feel free to browse through the different tabs of the experiment you're running and see how the metrics update as the experiment progresses. This tutorial does not include an explanation about these screens. 

---

## Commit the code changes

Let's commit our code to the repo. Go to your terminal and run the following commands:

```bash
$ git add .
$ git commit -m "integrate with missinglink"
$ git push
```

# Adding Resource Management

Now that we have everything working on our local workstation, let's take the integration to the next level. 

The next step is to run the experiment on a managed server. 
MissingLink can help you manage your servers, so that you don't have to worry about it.

For the sake of simplicity, we will not connect real GPU servers in this tutorial, but rather emulate a real server on our local workstation. This should definitely give you a sense of how it would work when running on real servers.

## The missing step

The most important step for setting up Resource Management in your project is to give us access to your training machines. To enable access, you will need to install MissingLink on your existing machines, or give us limited access to your cloud hosting account so we can spin up machines for you. As mentioned above, we will not perform this step in this tutorial.

## Let's emulate

For this step, take note of the name of our organization in the MissingLink system, and the ID of the **tutorials** project. Run the following command in the terminal:

```bash
$ ml projects list
```

You should get a list of all the projects you have access to. Look for the **tutorials** project:

```bash
project_id   | display_name | description | token        | org
<PROJECT_ID>   tutorials                    <SOME_TOKEN>   <YOUR_ORG>
```

Take note of the project ID of the **tutorials** project, as well as the name of the organization this project belongs to.

Now for some magic.

We'll run a command for launching the local server using the MissingLink CLI. Run the following in your terminal:

<!--- TODO: Remove params when possible  --->

```bash
$ ml run local --org <ORG_NAME> --project <PROJECT_ID> --git-repo git@github.com:<YOUR_GITHUB_USERNAME>/missinglink-tensorflow-tutorial1.git --cpu --image tensorflow/tensorflow:1.9.0-devel --command "python mnist_cnn.py"
```

This command takes the code you've committed to your forked repository, clones it to your local server, installs the requirements, and runs the expriment.  
Let's go over and explain all the options in the previous command:

`--org`: The name of the organization, taken from the previous step.  
`--project`: The id of the project, taken from the previous step.  
`--git-repo`: URL of a git that should be cloned to the server.  
`--cpu`: Use the CPU on your workstation. If your workstation has a supported GPU, you can specify `--gpu` instead.  
`--image`: The docker image to use. If your workstation has a supported GPU, you can skip this parameter to use the default docker image.
`--command`: The command to run (from the root of your source code).

---
**NOTE**

The command for running the same thing on a real server is very similar.

---

## Observe the progress

If everything goes well, we can now observe the progress of our experiment, running on a managed server, right in the dashboard.

Go to https://missinglink.ai/console and click the Resource Groups toolbar button on the left. You should see a newly created resource group representing our local emulated server. If you don't see it yet, the MissingLink CLI is still preparing it.

![Local resource group](./images/local_resource_group.png)

---
**NOTE**
  
This resource group is temporary and will disappear from the list once the job we're running is completed.

---

If you click the Jobs toolbar button on the left, you should see a newly created job.

![Job in jobs list](./images/jobs_list.png)

Click on the line showing the job. You are taken to a view of the logs of the task running in our local server.

![Job logs](./images/job_logs.png)

Let's see the actual progress of our experiment. Click the projects toolbar button on the left and choose the **tutorials** project. You should see the new experiment's progress.

# Summary

This tutorial demonstrated how you take an existing deep learning code sample, integrate the MissingLink SDK, and run it on an emulated local server.  

To learn more about what you can do with MissingLink, [head to the docs section](https://missinglink.ai/docs) on the MissingLink website.