Instructions to use VSM in a dedicated environment in your PC

1) Install Anaconda to use the package manager conda.

2) Create a new environment with ` conda create --name vsmenv ` where vsmenv is the name of your environment (can use any name)

3) Activate your environment with ` conda activate vsmenv ` 

4) Install the packages listed in requirements.txt with ` conda install -c conda-forge numpy pandas ...etc ` 
   or ` conda install -c conda-forge --file requirements.txt `

5) This is the environment that should be activated each time working with VSM with the instruction at point 3)

6) Launch VSM
6a) Jupyter Notebook - launch ` jupyter-lab ` and then open a notebook, e.g., ` VSM_writeinput.ipynb ` 
6b) Editor - may use ` spyder ` and open, e.g., ` VSM_test.py ` 
6c) Term - launch the python script as ` VSM_test.py ` 
