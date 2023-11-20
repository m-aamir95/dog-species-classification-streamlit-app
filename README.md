# Dog Species Classification App

## Table of Contents

- [Dog Species Classification](#sentiment-classification-app)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Fine-tuning the model](#Fine-tuning-the-model)
  - [Starting the Streamlit App which uses a fine-tuned model by default](#Starting-the-Streamlit-App-which-uses-a-fine-tuned-model-by-default)


## Description

The Dog Species Classification App is a powerful tool designed to classify 120 different dog breeds. Leveraging the Inception-V3 architecture as a pretrained model, our application has undergone fine-tuning using the 
[Stanford dog species datasets](http://vision.stanford.edu/aditya86/ImageNetDogs/) .


The frontend is powered by Streamlit, a Python library known for its simplicity in developing interactive and data-driven web applications. Streamlit facilitates rapid development with declarative syntax, ensuring an intuitive user interface.

The training codebase also integrates `Weights And Biases for MLOps`, which can be configured in the code. More on this in the setup [Installation and Application Startup](#Fine-tuning-the-model)


For simplified and swift deployments, `Docker` is utilized, enabling rapid and reproducible setups.

This repositoy can be divided into two parts.
1. Code for training and managing the model.
2. The streamlit App.



## Fine-tuning the model

To get started with the Project, follow these steps:

1. Clone the repository: `git clone [repository link]`
1. Navigate to the project directory: `cd dog-species-classification-streamlit-app`
1. Depending on your platform CPU or GPU install pytorch from here [Pytorch](https://pytorch.org/get-started/locally/)
1. For testing purposes you can use Google Colab or Kaggle as well, which comes with preconfigured pytorch dependencies.
1. Install other dependencies e.g, wandb for MLOps `sudo bash scripts/install_dependencies.sh`
1. The project uses WandB for experiment tracking. We need to login to WandB and provide our API token.
1. Initialize project configuration (if any) and login to WandB `sudo bash scripts/init_project_conf.sh` when prompted please enter your WandB key.
1. Download and configure the dataset `sudo bash scripts/download_and_setup_dataset.sh`
1. Various model training parameters have been defined in `sample.env` the codebase will read them via `.env` lets make a copy of sample.env `sudo cp sample.env .env`
1. Start the training `python train.py`

## Starting the Streamlit App which uses a fine-tuned model by default
1. Start all the containers `sudo docker-compose up -d --build`
1. The FastAPI is hosted at `PORT 8080` and the ReactJS based frontend is hosted at `PORT 80`
1. Nagivate to `http://localhost:80`
