import math
import random

import argparse

from missinglink import VanillaProject

parser = argparse.ArgumentParser()
parser.add_argument('--project')

args = parser.parse_args()

project = VanillaProject(project=args.project)

with project.experiment() as xp:
    logies = []
    with xp.train():
        for epoch in range(5):
            with xp.epoch(epoch):
                for i in range(0, 1000, 5):
                    val_loggy = math.log10(i * (1000 + epoch) + 1)
                    val_sin = math.sin(val_loggy)

                    loggy = val_loggy + math.log10(random.randint(1, i + 1))
                    sin = val_sin + math.log10(random.randint(1, i + 1))

                    logies.append(loggy)
                    with xp.batch(i + (1000 * epoch)):
                        xp.log_metric('loggy', loggy)
                        xp.log_metric('sin', sin)

                        if (i % 10) == 0:
                            with xp.validation():
                                xp.log_metric('loggy', val_loggy)
                                xp.log_metric('sin', val_sin)

    with xp.test() as test:
        test.set_labels(['cat', 'dog', 'frog'])

        xp.log_metric('adv_loggy', sum(logies) / len(logies))
        y_true = [2, 0, 2, 2, 0, 1]
        y_pred = [0, 0, 2, 2, 0, 2]
        test.add_test_data(y_true, y_pred)

        y_true = [2]
        y_pred = [0]
        test.add_test_data(y_true, y_pred)
