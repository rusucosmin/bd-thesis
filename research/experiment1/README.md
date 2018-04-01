# Experiment 1 - MNIST

Distillation for CNNs, trying different values for the
temperature parameter and different architectures.


## Pipeline

In the main file, main.py there is a pipeline for easy testing different architectures, training
both directly to MNIST and using distillation.


## Model tested and results

### Teacher model

Class name: "Teacher"

| # | Layer     | Input Shape | Output Shape | Description                                       |
|---|-----------|-------------|--------------|---------------------------------------------------|
| 1 | Input     | `N/A`       | `28x28x1`    |Input layer, `28x28 px` grayscale image            |
| 2 | Conv1     | `28x28x1`   | `28x28x32`   |First convolutional layer, `32` `5x5x1` convolution layers, with a relu activation function|
| 3 | MaxPool1  | `28x28x32`  | `14x14x32`   |First `2x2` max pooling layer|
| 4 | Conv2     | `14x14x32`  | `14x14x64`   |Second convolutiona layer, `64` `14x14x32` convolution layers, with a relu activation layer|
| 5 | MaxPool2  | `14x14x64`  | `7x7x64`     |Second `2x2` max pooling layer|
| 6 | FC        | `7x7x64`    | `1024`       |First fully connected layer, looks at all the nodes and outputs 1024 values|
| 7 | Dropput   | `1024`      | `1024`       |First dropout layer to avoid overfitting with a `p` hyperparameter|
| 8 | FC        | `1024`      | `10`         |Second fully connected layer - produces the logits|
| 9 | Softmax   | `10`        | `10`         |Softmax layer - produces the probability distribution of the classes|


### Student model

Class name: "Student"

| # | Layer     | Input Shape | Output Shape | Description                                       |
|---|-----------|-------------|--------------|---------------------------------------------------|
| 1 | Input     | `N/A`       | `28x28x1`    |Input layer, `28x28 px` grayscale image            |
| 2 | Conv1     | `28x28x1`   | `28x28x3`    |First convolutional layer, `3` `5x5x1` convolution layers, with a relu activation function|
| 3 | MaxPool1  | `28x28x3`   | `14x14x3`    |First `2x2` max pooling layer|
| 4 | Conv2     | `14x14x3`   | `14x14x6`    |Second convolutiona layer, `6` `14x14x3` convolution layers, with a relu activation layer|
| 5 | MaxPool2  | `14x14x6`   | `7x7x6`      |Second `2x2` max pooling layer|
| 6 | FC        | `7x7x6`     | `10`         |Fully connected layer, looks at all the nodes and outputs 10 logits|
| 7 | Softmax   | `10`        | `10`         |Last fully connected layer, produces the probability distribution of the classes|

### Student2 model

Class name: "Student2"

| # | Layer     | Input Shape | Output Shape | Description                                       |
|---|-----------|-------------|--------------|---------------------------------------------------|
| 1 | Input     | `N/A`       | `28x28x1`    |Input layer, `28x28 px` grayscale image            |
| 2 | Conv1     | `28x28x1`   | `28x28x3`    |First convolutional layer, `3` `5x5x1` convolution layers, with a relu activation function|
| 3 | MaxPool1  | `28x28x3`   | `14x14x3`    |First `2x2` max pooling layer|
| 4 | FC        | `14x14x3`   | `10`         |Fully connected layer, looks at all the nodes and outputs 10 logits|
| 5 | Softmax   | `10`        | `10`         |Last fully connected layer, produces the probability distribution of the classes|

### Student3 model

Class name: "Student3"

| # | Layer     | Input Shape | Output Shape | Description                                       |
|---|-----------|-------------|--------------|---------------------------------------------------|
| 1 | Input     | `N/A`       | `28x28x1`    |Input layer, `28x28 px` grayscale image            |
| 2 | Conv1     | `28x28x1`   | `28x28x2`    |First convolutional layer, `2` `5x5x1` convolution layers, with a relu activation function|
| 3 | MaxPool1  | `28x28x2`   | `14x14x2`    |First `2x2` max pooling layer|
| 4 | FC        | `14x14x2`   | `10`         |Fully connected layer, looks at all the nodes and outputs 10 logits|
| 5 | Softmax   | `10`        | `10`         |Last fully connected layer, produces the probability distribution of the classes|
