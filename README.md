# ML Internship 2023 - Image Classification Task

This repository contains the code for the Image Classification Task given for the ML Internship 2023. The task is to train a Machine Learning model on the EuroSAT land cover classification dataset.

## Dataset

The dataset used for this task is the EuroSAT land cover classification dataset. It contains 64x64 RGB images from Sentinel-2. Dataset link: [EuroSAT Dataset](https://github.com/phelber/EuroSAT)

## Repository Structure

This repository consists of two Jupyter notebooks:

1. **Training Notebook:** This notebook contains all the steps from loading the dataset, to training the model, saving the model, and evaluating its performance.

2. **Inference Notebook:** This notebook contains the code to perform inference, generate and save classification results using 20 sample images from the dataset.

## Instructions

1. Clone this repository to your local machine.
2. Ensure that you have Jupyter Notebook installed. If not, you can install it using `pip install notebook` or `conda install -c conda-forge notebook` if you're using Anaconda.
3. Navigate to the cloned repository and open the Jupyter notebooks to view the code and corresponding explanations.
4. To run the code, ensure that you have all the necessary Python packages installed. You can install the necessary packages using pip (e.g., `pip install numpy pandas keras tensorflow matplotlib seaborn scikit-learn`).
5. Before running the notebooks, please ensure that you have the correct path to the dataset. Modify the dataset paths in the notebooks as needed.

## Constraints of the Current Solution

-	Relative Low-level features of some image classes, such as Pasture and Highway categories, containing poor properties utilized to describe and understand the image (Corners, edges, angles, colours, etc...) which affect the recognition of their actual classes as long as their inﬂuence to the performance of deep neural networks in computer vision. 

-	ResNet50 Model is underfitting the training data since it performs poorly on the training data.

 
-	Very time-consuming operation for Fine-Tuning step because of the low performance of the CPU (Central Processing Unit), thus, we are obliged to decrease the number of epochs to avoid problems linked to processor or memory such as long processing time or memory errors.

-	Very small number of image samples for the inference (20 images) which leads to weak statistically significative results.

-	Repetitive Sample Data used of inference, since we have split our Dataset into only 2 sets (Training and Validation) --> Data redundancy which leads to eventually inaccurate conclusion.


-	Low Images Resolution (64x64 pixels): Could make the pre-trained model biased & effect of image resolution on validation set loss.

## Potential Improvements to the Solution

-	Increasing image resolution for pretrained CNN training (e.g., by SuperResolution with Enhancing High-Frequency Content): not necessarily improving performance can perhaps be likened to this similar phenomenon in which the higher parameter count presents an obstacle to performance not just owing to the risk of overfitting but also owing to the increased complexity of the optimization problem.

-	Improve CPU performance, in furtherance of trying other alternatives to obtain best significative image recognition outcomes, such as playing on the number of epochs until reaching the optimal value, number of batch size, changing the algorithm activation functions, learning rate value, etc.


-	Data Augmentation conductive to increase the diversity and amount of training data by applying random (but realistic) transformations. (e.g., Image resizes, Image rotation, Image flip, etc…)

-	With regards to inference, we define an independent subset from the dataset apart from training and validation which aimed to inference.


-	Increasing the number of sample images (referring to the Inference Notebook) 

## Dependencies

The code in this repository requires the following Python packages:

- Numpy
- Pandas
- Keras
- TensorFlow
- Matplotlib
- Seaborn
- Scikit-Learn

## Contact

If you encounter any issues or have questions, please open an issue in this GitHub repository.


