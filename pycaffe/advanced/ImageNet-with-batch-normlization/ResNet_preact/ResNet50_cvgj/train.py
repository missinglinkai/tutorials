import caffe
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
missinglink_callback.set_properties(display_name="ImageNet-ResNet50 batch norm")
missinglink_callback.set_monitored_blobs(["loss"])
solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, 'train.solver')

solver.solve()
