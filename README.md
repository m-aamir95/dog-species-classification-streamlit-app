# Dog Species Classification App

## Table of Contents

- [Dog Species Classification](#sentiment-classification-app)
  - [Table of Contents](#table-of-contents)
  - [Description](#description)
  - [Installation and Application Startup](#installation-and-application-startup)
  - [Usage](#usage)


## Description

The Dog Species Classification App is a powerful tool designed to classify 120 different dog breeds. Leveraging the Inception-V3 architecture as a pretrained model, our application has undergone fine-tuning using the 
[Stanford dog species datasets](http://vision.stanford.edu/aditya86/ImageNetDogs/) .


The frontend is powered by Streamlit, a Python library known for its simplicity in developing interactive and data-driven web applications. Streamlit facilitates rapid development with declarative syntax, ensuring an intuitive user interface.



For simplified and swift deployments, Docker is utilized, enabling rapid and reproducible setups.


## Installation and Application Startup

To get started with the Project, follow these steps:

1. Clone the repository: `git clone [repository link]`
2. Navigate to the project directory: `cd fastapi-ml-service`
3. Install the linux dependencies including docker: `sudo bash bash_scripts/install_system_dependencies.sh`
4. Currently we need to manually create the `sentiment_db` inside the mysql docker container (Ideally it should be a part of post container init script). Following are the steps to manually create the `sentiment_db` inside the mysql container.
   *  `sudo docker-compose up --build -d db`
   *  `sudo docker exec -it db bash` # Jump into the db container
   *  `mysql -u root -p`
   *  When prompted enter the mysql root password which is by default set to `my-secret-pw`
   *  `create database sentiment_db;`
   *  exit # Exit MySQL shell
   *  exit # Exit container
5. Start all the containers `sudo docker-compose up -d --build`
6. The FastAPI is hosted at `PORT 8080` and the ReactJS based frontend is hosted at `PORT 80`
7. Nagivate to `http://localhost:80`

## Usage

1. The FastAPI API points can be viewed by visiting http://localhost:8080/docs
2. 
3. You can also interact with the app by visiting http://localhost:80 