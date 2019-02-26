'''
This script should be run from the caffe root directory.
Please see instructions link for details!
'''

import caffe
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)

missinglink_callback.set_properties(display_name="Caffe Net Flickr Style")

solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, 'models/finetune_flickr_style/solver.prototxt')

missinglink_callback.set_expected_predictions_layers("label", "fc8_flickr")

missinglink_callback.set_monitored_blobs(["loss", "accuracy"])

solver.solve()
