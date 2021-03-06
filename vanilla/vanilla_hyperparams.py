import math
import random

import argparse

from missinglink import VanillaProject


parser = argparse.ArgumentParser()
parser.add_argument('--project')

args = parser.parse_args()

project = VanillaProject(project=args.project)

project.set_hyperparams(
    LEARNING_RATE=0.01,
)

with project.experiment() as experiment:
    loggies = []
    with experiment.train():
        for i in experiment.loop(1000):
            val_loggy = math.log10(i + 1)
            val_sin = math.sin(val_loggy)

            loggy = val_loggy + math.log10(random.randint(1, i + 1))
            sin = val_sin + math.log10(random.randint(1, i + 1))

            loggies.append(loggy)
            experiment.log_metric('loggy', loggy)
            experiment.log_metric('sin', sin)

            if (i % 10) == 0:
                with experiment.validation():
                    experiment.log_metric('loggy', val_loggy)
                    experiment.log_metric('sin', val_sin)

    with experiment.test() as test:
        test.set_labels(['cat', 'dog', 'frog'])

        experiment.log_metric('adv_loggy', sum(loggies) / len(loggies))
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        test.add_test_data(y_true, y_pred)

        y_true = [2]
        y_pred = [0]
        test.add_test_data(y_true, y_pred)
