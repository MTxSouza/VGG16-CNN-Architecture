# **VGG16-CNN-Architecture**
## **Info**
### **- library**
Folder which contains all necessary code to run the main application.
### **- test**
Folder responsible to test if all requirement packages was installed.
### **- setup.py**
Script which installs all necessaries packages.
## **Setup**
**(Recommended): Create a virtual enviroment to run all cells below. On terminal, type the command below.*
```
python3 -m venv env # This will create a virtual enviroment with name 'env'.
source env/bin/activate # Activate the virtual enviroment to be used.
```
First, open the terminal and run the cell below to setup the enviroment. This script will install all necessaries packages to run the main application.

*Make sure  you are inside of the project folder.*
```
pip install .
```
After that, you can verify if all packages was installed correctly running the command below. (Not working yet)
```
python3 test/run.py
```
## **Run**
To run the main application (the training step), run the command below.
```
python3 library/train.py
```
*There are some arguments you can pass before start training.*
>--epochs (int>1 | default=10): Number of iterations for each batch in your data.

>--batch (int>=1 | default=32): Split your data to be passed into neural net.

>--name (str | default='vgg16'): Final name of your trained model.
```
python3 library/train.py --epochs 100 --batch 16 --name my_model
```
