# Pairwise-Compatibility-Prediction-using-a-MobileNetV2
Pairwise compatibility prediction of polyvore outfits dataset using CNN built from scratch and a fine tuned MobileNetV2 model

# Dataset description
Polyvore Outﬁts is a real-world dataset created based on users’ preferences of outﬁt conﬁgurations on an online website named polyvore.com: items within the outﬁts that receive high-ratings are considered compatible and vice versa. It contains a total of 365,054 items and 68,306 outﬁts. The maximum number of items per outﬁt is 19. A visualization of an outﬁt is shown in Figure 1. The images and json files are available at: https://drive.google.com/file/d/1ZCDRRh4wrYDq0O3FOptSlBpQFoe0zHTw/view?usp=sharing

# File description
       dataloader.py: Data preprocessing and loading.
       model.py: MobileNetV2 and a finetuned model
       utils.py: hyperparameters and file paths

# To install the complete list of dependencies, run:
       pip install -r requirements.txt

# Running the code:
Set the parameters in utils.py. The code uses CUDA which needs an Nvidia GPU. If not using a GPU, set use_cuda flag to False in utils.py.

# References
       https://github.com/davidsonic/EE599-CV-Project
