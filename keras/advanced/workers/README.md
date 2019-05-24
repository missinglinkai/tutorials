```diff
 from keras.layers import Conv2D, MaxPooling2D
 from keras.utils import Sequence
 import numpy as np
 
+import missinglink
+
+missinglink_callback = missinglink.KerasCallback()
+
 workers = 8
 batch_size = 128
 num_classes = 10
 epochs = 1
 # input image dimensions
 img_rows, img_cols = 28, 28
 
+missinglink_callback.set_hyperparams(workers=workers)
 
# ...
 
 model.fit_generator(
     SeqGen(x_train, y_train, batch_size=batch_size),
     epochs=epochs,
     verbose=1,
     validation_data=(x_test, y_test),
-    callbacks=[],
+    callbacks=[missinglink_callback],
     workers=workers,
 )
 
 print("Evaluating")
 
 score = model.evaluate(x_test, y_test, verbose=0)
```
