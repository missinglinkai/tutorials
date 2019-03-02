import caffe
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)

missinglink_callback.set_properties(display_name="Gender Deep Learning")

solver = missinglink_callback.create_wrapped_solver(
    caffe.SGDSolver,
    'solver_test_fold_is_0.prototxt'
)

missinglink_callback.set_expected_predictions_layers("label", "fc8")

solver.solve()
