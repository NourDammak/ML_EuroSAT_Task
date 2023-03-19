**Constraints:**

- Relative Low-level features of some image classes, such as Pasture and Highway categories, containing poor properties utilized to describe and understand the image (Corners, edges, angles, colours, etc...) which affect the recognition of their actual classes as long as their inﬂuence to the performance of deep neural networks in computer vision. 

- ResNet50 Model is underfitting the training data since it performs poorly on the training data.



- Very time-consuming operation for Fine-Tuning step because of the low performance of the CPU (Central Processing Unit), thus, we are obliged to decrease the number of epochs to avoid problems linked to processor or memory such as long processing time or memory errors.

- Very small number of image samples for the inference (20 images) which leads to weak statistically significative results.

- Repetitive Sample Data used of inference, since we have split our Dataset into only 2 sets (Training and Validation) à Data redundancy which leads to eventually inaccurate conclusion.


- Low Images Resolution (64x64 pixels): Could make the pre-trained model biased & effect of image resolution on validation set loss.

**Potential improvements to the solution:**

- Increasing image resolution for pretrained CNN training (e.g., by SuperResolution with Enhancing High-Frequency Content): not necessarily improving performance can perhaps be likened to this similar phenomenon in which the higher parameter count presents an obstacle to performance not just owing to the risk of overfitting but also owing to the increased complexity of the optimization problem.

- Improve CPU performance, in furtherance of trying other alternatives to obtain best significative image recognition outcomes, such as playing on the number of epochs until reaching the optimal value, number of batch size, changing the algorithm activation functions, learning rate value, etc.


- Data Augmentation conductive to increase the diversity and amount of training data by applying random (but realistic) transformations. (e.g., Image resizes, Image rotation, Image flip, etc…)

- With regards to inference, we define an independent subset from the dataset apart from training and validation which aimed to inference.


- Increasing the number of sample images (referring to the Inference Notebook) 
