import caffe
import missinglink

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)

solver = missinglink_callback.create_wrapped_solver(
    caffe.SGDSolver,
    "poolmean_solver.prototxt"
)

missinglink_callback.set_properties(display_name="Videos 2 Language")

missinglink_callback.set_monitored_blobs(["softmax_loss", "accuracy"])

caffe.set_mode_cpu()

solver.solve()
