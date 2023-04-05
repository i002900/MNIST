## Using MNIST from Keras
from keras.datasets import mnist
(train_images , train_labels) , (test_images , test_labels ) = mnist.load_data()
## Flatten the images. For dxample, 28 * 28 to 1 * 784
flatten = train_images.shape[1] * train_images.shape[2]
X_train = train_images.reshape(len(train_images), flatten )
X_test = test_images.reshape(len(test_images), flatten )

## Grey Scale is 0 to 255. Divide by 255 to normalize values 0 to 1.
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

## Import K Nearest Neighbour algorithm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

## Consider 3 nearest neighbours for classification and build the model 
neigh = KNeighborsClassifier(n_neighbors=3)

## Train the model by fitting the training set
neigh.fit(X_train, train_labels)

# Predict the Testing set
y_neigh = neigh.predict(X_test)

# Calculate the accuracy
accuracy_kNN = accuracy_score(y_neigh, test_labels)
print(f"The accuracy using kNN with 3 neighbors is: {accuracy_kNN}")

## The observed accuracy is 97.05%
