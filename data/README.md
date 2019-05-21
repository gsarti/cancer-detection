# Data

Data is divided in `train` and `test` sets, dowloadable from the following links given their size.

The [training set](https://www.kaggle.com/c/histopathologic-cancer-detection/download/train.zip) contains roughly 220k images of 96x96px identified by a unique `id` as filename. The histopathological images are glass slide microscope images of lymph nodes that are stained with **hematoxylin and eosin (H&E)**. This staining method is one of the most widely used in medical diagnosis and it produces blue, violet and red colors. Dark blue hematoxylin binds to negatively charged substances such as nucleic acids and pink eosin to positively charged substances like amino-acid side chains (most proteins). Typically nuclei are stained blue, whereas cytoplasm and extracellular parts in various shades of pink.

The [test set](https://www.kaggle.com/c/histopathologic-cancer-detection/download/test.zip) contains roughly 57.5k images for which the label should be predicted by our system.

The file `train_labels.csv` contains the `id` for each image alongside with their true `label` for training purposes. The label is 1 when the central 32x32px region of the image contains at least one pixel of tumor tissue, or 0 otherwise. Tumor tissue in the outer region of the patch does not influence the label. This outer region is provided to enable fully-convolutional models that do not use zero-padding, to ensure consistent behavior when applied to a whole-slide image.