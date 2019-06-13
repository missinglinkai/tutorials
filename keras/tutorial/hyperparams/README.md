```diff
missinglink_callback.set_properties(
     display_name='Keras convolutional neural network',
     description='Two dimensional convolutional neural network')
 
+missinglink_callback.set_hyperparams(
+    conv_dropout=CONV_DROPOUT)
+
model.fit(
     x_train, y_train, batch_size=BATCH_SIZE,
     nb_epoch=EPOCHS, validation_split=VALIDATION_SPLIT,
     callbacks=[missinglink_callback])

```