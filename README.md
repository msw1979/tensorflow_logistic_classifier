# Logistic Regression Using TensorFlow
This is an example of Logistic Regression using TensorFlow. The data used here is the tumor data and can be downloaded from: https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data. The code split the data to training and validation data sets and normalize the feature part, then load it to training loaders. The user has the choice to use the custom class model and sequential model. The code will generate loss and accuracy data and plot them. It also plot confusion matrix plots for training and validation data. 
The figure below show the loss and accuracy versus epoch for training and validation process:

Custom Class Model:

![loss_accuracy_epoch_class_model](https://user-images.githubusercontent.com/12114448/236636722-4273cbcc-7a6f-437e-b2d5-f635e2c884ee.png)


Sequential Model:

![loss_accuracy_epoch_seq_model](https://user-images.githubusercontent.com/12114448/236636731-e106a57a-327d-41c9-b624-0184e8245de5.png)

The figures below show the confusion matrix for training and validation data:

Custom Class Model:

![confusion_matrix_test_class](https://user-images.githubusercontent.com/12114448/236636745-0b609a42-7453-4384-b846-5843cdcebe16.png)


Sequential Model:

![confusion_matrix_test_seq](https://user-images.githubusercontent.com/12114448/236636749-75896c2b-6ac9-4701-a25d-5a2c573cd659.png)
