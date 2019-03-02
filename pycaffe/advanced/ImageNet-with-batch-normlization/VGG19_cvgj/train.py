import caffe
import missinglink

caffe.set_mode_cpu()
missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
missinglink_callback.set_properties(display_name="ImageNet-VGG16 batch norm.")

solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, 'train.solver')

missinglink_callback.set_expected_predictions_layers("label", "fc8")
missinglink_callback.set_monitored_blobs(["loss", "accuracy"])

solver.solve()
