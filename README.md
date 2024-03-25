# Traffic rule violation detecting project with OpenCV and PyTorch

In some countries, semi-trucks are not allowed to use the inner lane on motorways during daytime. This project detects this situation and saves the evidence in a PostgreSQL database so the public authorities can initiate proceedings against the offender.
In this project, I've built and trained a deep neural convolutional network to classify vehicles using PyTorch. OpenCV is used to process and filter frames and to localise moving objects on video.

## Dataset
CNN model is trained on the dataset of [IEEE BigData Cup Challenge 2022](https://github.com/sekilab/VehicleOrientationDataset). The images are prepared for object detection problems so I've cut every object to make it into an object classification dataset. The project uses the car_front and truck_front images of the dataset.
