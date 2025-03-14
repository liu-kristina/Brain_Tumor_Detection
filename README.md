# Brain Tumor MRI Classification Using Pre-trained Models

## What is a brain tumor?

A brain tumor is a collection, or mass, of abnormal cells in your brain. Your skull, which encloses your brain, is very rigid. Any growth inside such a restricted space can cause problems. Brain tumors can be cancerous (malignant) or noncancerous (benign). When benign or malignant tumors grow, they can cause the pressure inside your skull to increase. This can cause brain damage, and it can be life-threatening.

## The importance of the subject

Early detection and classification of brain tumors is an important research domain in the field of medical imaging and accordingly helps in selecting the most convenient treatment method to save patients life therefore.

## Dataset

The dataset was taken from [here](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset).

### Dataset Details

This dataset contains **7022** images of human brain MRI images which are classified into 4 classes:

- glioma
- meningioma
- no tumor
- pituitary

About 22% of the images are intended for model testing and the rest for model training. These notebooks were run in Google Colab.


## Pre-trained Model

A pre-trained model is a model that was trained on a large benchmark dataset to solve a problem similar to the one that we want to solve. Accordingly, due to the computational cost of training such models, it is common practice to import and use models from published literature (e.g. VGG, Inception, ResNet50). For this project, I decided to use **VGG19** model to perform image classification for brain tumor MRI images.[VGG19 Article](https://arxiv.org/abs/1409.1556)
