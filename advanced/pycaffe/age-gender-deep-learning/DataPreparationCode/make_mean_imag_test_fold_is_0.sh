TOOLS=/opt/caffe/build/tools
DATA=AdienceFaces/lmdb/Test_fold_is_0/gender_train_lmdb
OUT=AdienceFaces/mean_image/Test_folder_is_0

$TOOLS/compute_image_mean $DATA $OUT/mean.binaryproto

