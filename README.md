# Deep Learning Emotion Classifier

This project is a deep-learning model for image classification. The model is a convolutional neural network built using TensorFlow, which can classify images into two classes: 'Happy' or 'Sad'. The model is trained, validated and tested using image datasets that are preprocessed before use.

## Project Organization

The project is divided into three primary Python scripts:

1. `data_preprocessing.py`: Contains the logic for preprocessing image data, which includes reading images, checking for valid file extensions, resizing, and normalizing pixel values. The data is also split into training, validation, and testing sets in this step.

2. `model_training.py`: Defines the architecture of the Convolutional Neural Network (CNN) used for classification. It consists of several convolutional, max-pooling, flattening and dense layers. The model is compiled and trained using the training and validation sets prepared in the preprocessing step.

3. `evaluate_test.py`: Evaluates the model's performance on the testing set. It also includes functions for predicting the class of a single image, saving the trained model, and loading the saved model for future use.

In addition to these, a `main.py` script is provided to tie all the steps together and run the entire pipeline from preprocessing to prediction.

## Usage

To use this project, follow these steps:

1. Clone the repository to your local machine. (Vitual Env is Optional)
2. Ensure that you have the necessary Python libraries installed. These include TensorFlow, OpenCV, Matplotlib, and Numpy. You can install them with pip:   
   
```bash
pip install tensorflow opencv-python-headless matplotlib numpy
```  
3. Place your image data in a directory named 'data'. The images should be divided into two subdirectories, one for each class ('Happy' and 'Sad').   
   I would say the pic and spec of the imgs don't matter in this project since we are rescaling it during preprocessing.  

5. Run the main.py script with:
```bash
python main.py
```
This will preprocess the data, train the model, evaluate its performance, and save the model.

You can visualize the first batch of preprocessed images using the visualize_data() function from data_preprocessing.py.

To predict the class of a single image, use the predict_class() function from evaluate_test.py.  


