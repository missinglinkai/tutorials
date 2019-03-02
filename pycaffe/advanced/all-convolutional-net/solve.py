import caffe
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
missinglink_callback.set_properties(display_name="All CNN")

solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, "ALL_CNN_C_solver.prototxt")

missinglink_callback.set_expected_predictions_layers("label", "pool")

solver.solve()
