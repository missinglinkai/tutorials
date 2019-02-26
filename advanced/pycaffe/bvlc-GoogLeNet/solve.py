import caffe
import missinglink

caffe.set_mode_cpu()

missinglink_callback = missinglink.PyCaffeCallback(
    owner_id="OWNER_ID",
    project_token="PROJECT_TOKEN"
)
missinglink_callback.set_properties(display_name="BVLC GoogLeNet")

solver = missinglink_callback.create_wrapped_solver(caffe.SGDSolver, 'solver.prototxt')
missinglink_callback.set_expected_predictions_layers("label", "loss1/classifier")
missinglink_callback.set_monitored_blobs(["loss1_loss1", "loss1_top-1", "loss1_top-5",
                                          "loss2_loss2", "loss2_top-1", "loss2_top-5",
                                          "loss3_loss3", "loss3_top-1", "loss3_top-5"])

solver.solve()
