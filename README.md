# Cancer Detection from CT Scans

## Motivation

Lung cancer is one of the deadliest cancer types in the world, responsible for a large number of
cancer-related deaths each year. It’s crucial to detect lung cancer early to improve survival rates.
Treatment options are far more effective when the disease is caught in its early stages.

Right now, the diagnostic process relies on radiologists manually examining images from chest
CT scans. This approach is not only slow and labor-intensive but also prone to human error. The
complexity of medical images, variability in image quality, and differences in radiologists’ experience
make it tough to guarantee consistent and timely diagnoses. These challenges show that we need a
better, more efficient, and scalable solution for lung cancer detection.

To address this problem, our team is working on an automated lung cancer detection model using
convolutional neural networks (CNNs). Using deep learning techniques, we can process more chest
CT scan images faster, and more accurately, than traditional manual analysis.
This solution can automate lung cancer screening, reduce human error, and help solve some of the key challenges in the diagnostic process.

## Data Understanding

We used two datasets for this project. Both consist of CT scan images of the lungs. The first dataset,
dataset 1, is split up into two categories: cancer and normal images. The second dataset, dataset
2, has 4 categories: adenocarcinoma, squamous cell carcinoma, large cell carcinoma, and normal
images, in which the first three are different types of lung cancer. See *figure 1* for example images
from dataset 1 and *figure 2* for images from dataset 2. All images are grayscale and the sizes vary
between 200 and 1200 pixels in width and between 200 and 850 pixels in height.

<p align="center">
<img width="476" alt="Screenshot 2025-03-24 at 11 42 55 PM" src="https://github.com/user-attachments/assets/677fe40c-cbf6-46fb-8ed0-028fc21c0e9e" />

<p align="center">
<img width="581" alt="Screenshot 2025-03-24 at 11 43 18 PM" src="https://github.com/user-attachments/assets/63c1c5a2-632c-4e93-af08-76d705dd12a7" />

In dataset 1, the severity of the tumor varied. The cancer image in *figure 1* is an image with a
relatively large and obvious tumor compared to many other images. Looking at *figure 2*, it is easy to
distinguish between certain classes and harder to see any differences between others. For example,
squamous carcinoma and the normal images are easy to tell apart whereas it is not as easy to see a
difference between the adenocarcinoma and large cell carcinoma images. 

In *table 1*, we see the size of the datasets and the sample distributions across training, validation,
and test sets. Although neither dataset is particularly large, dataset 1 is roughly 4.5 times bigger
than dataset 2.

<p align="center">
<img width="462" alt="Screenshot 2025-03-24 at 11 44 22 PM" src="https://github.com/user-attachments/assets/a03cbf6c-4912-43fd-ab01-8cf3e4e3ec2d" />

The distribution of the different classes in both datasets is shown in *figure 3* across the train
and test data. We observe that there are almost twice as many cancer images in the training
data for dataset 1 whereas they are more evenly distributed for the test data. For dataset 2, the adenocarcinoma is the most common class and large cell carcinoma is the class with the fewest images.

<p align="center">
<img width="492" alt="Screenshot 2025-03-24 at 11 44 51 PM" src="https://github.com/user-attachments/assets/665b9c42-6114-4918-900a-6b2e9a3a2944" />

## Data Preparation

The data preparation was straight forward for both datasets. All images went through the same
transformation which involved normalizing the pixel values, resizing them to the same size (224,
224), and converting them to tensors. The shape of the tensors after the transformation was (1,224, 224). For dataset 2, we randomly rotated the images 15 degrees and randomly flipped the images horizontally. This was done to increase the difficulty of the dataset which would improve the
robustness of the model and hence increase its performance on the test data. This was particularly
important because of the small size of the dataset.

After transforming the data we loaded the data using batch size 128 for dataset 1 and batch
size 8 for dataset 2 since it is a smaller dataset. We also made a train/validation split with 80% for
training and 20% for validation. The test data was in a separate file so we did not need to create a
test set.

## Modeling

### Dataset 1

We constructed a Convolutional Neural Network (CNN) designed to classify images into two categories: Cancer and Normal. The model combined the convolutional layers, used for feature extraction, with fully connected layers used for classification. Given that we were working with grayscaled
images, the convolutional layers began with 1 input channel. This then expanded to 16, 32, and
64 channels, respectively, across 3 layers. Each layer used a kernel size of 3x3, with padding to
preserve spatial dimensions. We implemented a pooling layer using MaxPool2d to reduce the spatial
dimensions by a factor of 2. Then, the first fully connected layer transformed the newly flattened
feature map into 500 neurons, utilizing a dropout rate of 25% applied to its output. The second fully
connected layer mapped these 500 input features to the two output neurons for classification. We
also attempted to use deeper and more complex models such as VGG16. However, once we realized
that the deeper models performed worse, we went back to our simpler model.

### Dataset 2

Given that we were working with a smaller dataset, we utilized a pre-trained model, namely
ResNet18, and modified it to best perform our classification task. This allowed us to leverage
the features learned from a large dataset, and reduce the need for increased training data. We first
modified the model by changing the input from 3 channels to 1, indicative of our use of grayscaled
images. We then replaced the original fully connected layer with 3 linear layers of our own. The
first layer transformed the 512 input features into 512 output features, with a 30% dropout rate to
reduce overfitting. This was followed by a second linear layer, reducing the number of features from
512 to 256, with a 20% dropout rate used to further improve generalization. Finally, the third layer
transformed those 256 features into our 4 output classes.

### Implementation

In training and evaluating our models, we applied the cross entropy loss function to tackle our
classification problem. We trained and validated over epochs and with early stoppage occurring after a number of consecutive validation losses were not below the minimum values. The stoppage
was triggered after 2 consecutive values for dataset 1, and 3 consecutive values for dataset 2. We
implemented this strategy to train over as many epochs as possible before the model began overfitting. We tried multiple optimizers and found that Adam performed best, and resulted in fast
convergence. For the dataset 2 model, we found that Adam with weight decay performed better due
to it helping to prevent overfitting. For both models we calculated the testing accuracy overall, and
among specific categories. This was especially useful for our second dataset to distinguish where our
model performed strongly and weakly.

## Results and Evaluation

### Dataset 1
Our model achieved an overall test accuracy of 100%. Both training and validation loss were tracked
during training, with each steadily decreasing before flattening out at epoch 5. The validation loss
continued to decrease incrementally until epoch 20 without triggering an early stoppage. We achieved
a minimum validation loss of 0.000030. We determined that the test accuracy would be the most
important metric given the nature of the task. Given the 100% accuracy for the testing data, we
are extremely confident the model can distinguish between cancerous and non-cancerous scans. As
such, it can be an extremely useful tool to screen CT-Scans for lung cancer, and save valuable time
for doctors. 

<p align="center">
<img width="406" alt="Screenshot 2025-03-24 at 11 51 07 PM" src="https://github.com/user-attachments/assets/75d7771d-4fa2-4bf5-9ab4-fdd5d91608d0" />

<p align="center">
<img width="454" alt="Screenshot 2025-03-24 at 11 51 30 PM" src="https://github.com/user-attachments/assets/8dfb5ca9-da84-4378-8422-968428b5ebf5" />

### Dataset 2
Our model achieved an overall test accuracy of 84%. By category, its accuracy was
70% for adenocarcinoma, 98% for large cell carcinoma, 98% for normal, and 86% for squamous
cell carcinoma. Both training and validation loss were tracked during training, with each generally
decreasing until epoch 15 where the validation loss began to stabilize. Early stopping was triggered
at epoch 19 after 3 consecutive validation losses that were not below the minimum value of 0.40.

Similarly to dataset 1, our key indicator of performance is measured by test accuracy. The model performed exceptionally well for large cell carcinoma and normal categories, with weaker
performance for adenocarcinoma and squamous cell carcinoma. It is worth noting that the model
never predicted normal when the patient had a form of cancer. This is very important given the
correlation between early detection and survival rates.
Although the model can be improved with more training data, it can already effectively be used
for initial screening of CT-Scans to determine if further analysis is required. Given the high accuracy
rate, the model can help to automate the detection process, and provide initial analysis for facilities
lacking the expertise needed to diagnose.

<p align="center">
<img width="446" alt="Screenshot 2025-03-24 at 11 53 33 PM" src="https://github.com/user-attachments/assets/35207fcd-bf41-4742-a4f9-0e99ad3c65cc" />

<p align="center">
<img width="559" alt="Screenshot 2025-03-24 at 11 53 01 PM" src="https://github.com/user-attachments/assets/ae41681f-8aa4-4849-aa59-cf96e24ec60a" />

## Deployment

The main aim is to deploy the model as a diagnostic tool for radiologists, automating the lung
cancer detection from chest CT scans. This deployment will improve diagnostic accuracy, streamline
workflows, and help prioritize high-risk cases for quick medical attention. The model can be licensed
to healthcare providers and research institutions which will serve as the main revenue driver for this
product.

## References

*Mohamed Hany. Kaggle Chest CT-Scan. https://www.kaggle.com/datasets/mohamedhanyyy/
chest-ctscan-images.*

*Faizan Khan. Kaggle Lung C. https://www.kaggle.com/datasets/faizankhan20cab105/
lung-c/data.*

## Contributors

*Ryan Gilbert*

*Emil Westling*

*Yurou Xu*

*Chelsy Chen*

*Junze Cao*










