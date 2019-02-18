import time

import tensorflow as tf
import missinglink

NUM_EPOCHS = 4
NUM_BATCHES = 10

missinglink_project = missinglink.TensorFlowProject(project_token='KzcbCxZWjewiqxCi')

# Build a graph.
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b


with missinglink_project.create_experiment() as experiment:
    for epoch in experiment.epoch_loop(NUM_EPOCHS):
        for batch in experiment.batch_loop(NUM_BATCHES):
            time.sleep(0.5)
            with experiment.train():
                # Launch the graph in a session.
                sess = tf.Session()

                # Evaluate the tensor `c`.
                print('sess run result', sess.run(c))
                loss = batch * 0.1 - epoch * 0.05
                print('loss', loss)
                experiment.add_metric('loss', loss)


