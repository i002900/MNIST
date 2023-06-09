The MNIST (Modified National Institute of Standards and Trechnology) dataset is a curated list of handwritten digits by a mix of American census bureau employees and high school students. 
The images in this dataset were normalized to fit-in a 28 by 28 pixel bounding box and in gray scale. This is a popular data set and is often used when training models for numerical character recognition.
A simple implementation of the KNearest Neighbour algorithm was used to build the model.

What makes this project unique is, that I have used the model to decipher my own handwritten digits - by scanning my handwritten digits using a home scanner and some of them created using Paint. Initially, the model failed miserably. After much frustration I realized that the model wass trained using properly centred 28 X 28 pixels in grey scale! This required additional code to transform my personally scanned images to the format of the MNIST dataset that was used for training the model. (Transformation Code kind courtesy Ole Kroger).
Testing the Model:
Display the figure in position 1511 of the test data set:
<img src = https://github.com/i002900/MNIST/blob/main/Figure%201511.png>

Also print the predicted value in position 1511.
You will see the predicted value is also 4.
