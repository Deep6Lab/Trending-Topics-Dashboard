# An Interactive Web-Based Dashboard to Track Trending Topics in Financial Journals
This project builds a web-based interactive dashboard to detect trends in financial journals. The dashboard can be modified to detect trends for other journals.

## Technologies

- Flask 0.12 or later
- Python 3.8 or later
- Python libraries: numpy, pandas, NLTK, sklearn, scipy, matplotlib, plotly, jupyter_dash, dash, wordcloud
- [Anaconda Navigator](https://anaconda.org/anaconda/anaconda-navigator)

## Installation Guide for Setting Up & Running the App

### Environment Setup for Windows

 1. Download and install [Anaconda Navigator](https://www.anaconda.com/products/individual)
 1. Create a virtual environment for the project following these steps:
    - Open __Anaconda Navigator__
    - Click on __Environment__ tab on the left panel
    - Click on the ___Create (+)___ button at the bottom left corner of the screen
    - Fill in the information on the dialog box and click the ___Create___ button
    - <img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/anaconda_env.PNG" width="700px" height="500px" />
 1. Click on the __Home__ tab on the left panel of __Anaconda__ and do the followings:
    - Click on the ___Install___ button under __Jupyter Notebook__ to install Jupyter Notebook
    - Click on the ___Install___ button under __CMD.exe Prompt__ to install the command prompt for current environment 
 
### Application Setup and Startup
 
#### Install Required Libraries
 1. Clone this repository and save to the project folder
 1. Install the required libraries following these steps:
    - In __Anaconda > Home__, open __Command prompt__ by clicking the ___Launch___ button under __CMD.exe Prompt__ for the current environment
    - Go to project directory and type the following command in the Command prompt:<br /> __pip install -r requirements.txt__
    - <img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/install_command.PNG" width="700px" height="45px" />
 
#### Start & Run the App
 
 1. In __Anaconda > Home__, click on the ___Launch___ button under __Jupyter Notebook__ to open Jupyter Notebook in the web browser.
 1. Open and run __[05_dashboard.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/05_dashboard.ipynb)__ notebook. Then click on the URL output from the last cell to launch the app. **Note:** Sometimes the port number might already be running on your system. In that case change the port numerb to 5000 or 7000.
    - <img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/dash_app.PNG" width="600px" height="170px" />

### Environment Setup for Mac

 1. Download and install [Anaconda Navigator](https://www.anaconda.com/products/individual)
 1. Create a virtual environment for the project following these steps:
    - Open __Anaconda Navigator__
    - Click on __Environment__ tab on the left panel
    - Click on the ___Create (+)___ button at the bottom left corner of the screen
    - Fill in the information on the dialog box and click the ___Create___ button
    - <img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/anaconda_env.PNG" width="700px" height="500px" />
 1. Click on the __Home__ tab on the left panel of __Anaconda__ and do the followings:
    - Click on the ___Install___ button under __Jupyter Notebook__ to install Jupyter Notebook
    - Click on the ___Launch___ button under __Jupyter Notebook__ to launch Jupyter Notebook
    - Once the application opens on the web browser, click on __New__ and then select __Terminal__
    - <img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/Terminal_Screenshot.png" width="600px" height="170px" />
    - Inside the terminal, run __conda activate trend_dashboard__
 
### Application Setup and Startup
 
#### Install Required Libraries
 1. In the same terminal, 
    - Clone this repository using the command __git clone https://github.com/Deep6Lab/trending-topics-dashboard.git__
    - Update the path using the command __cd uidashboard__
    - Then run the command __pip install -r requirements.txt__
 
#### Start & Run the App
 
 1. In __Anaconda > Home__, click on the ___Launch___ button under __Jupyter Notebook__ to open Jupyter Notebook in the web browser.
 2. Locate and open the folder titled "uidashboard".
 3. Open and run __[05_dashboard.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/05_dashboard.ipynb)__ notebook. Then click on the URL output from the last cell to launch the app. **Note:** Sometimes the port number might already be running on your system. In that case change the port numerb to 5000 or 7000.
    - <img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/dash_app.PNG" width="600px" height="170px" />
    
## Screenshots of Application

#### Journal Trends: Simple UI

<img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/dashboard_front_page.PNG" width="1000px" height="600px" />

#### Journal Trends: Advanced UI

<img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/advanced_ui.PNG" width="1000px" height="600px" />

#### Journal Trends: Diagnostics

<img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/diagnostics.PNG" width="1000px" height="400px" />

#### Dataset

<img src="https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/images/dataset.PNG" width="1000px" height="600px" />


## App Development Guide for Developers

To build the app on ___new dataset___ or rebuild the app on the ___updated dataset___, run the following Jupyter notebooks in the order listed:

1. Data preprocessing: [00_data_preprocessing.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/00_data_preprocessing.ipynb)
1. Feature extraction: [01_features_extraction.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/01_features_extraction.ipynb)
1. Data Normalization/Reduction: [02_feature_engineering.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/02_feature_engineering.ipynb)
1. Model Results: [04_model_results.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/04_model_results.ipynb)
1. Interactive Dashboard: [05_dashboard.ipynb](https://github.com/Deep6Lab/trending-topics-dashboard/blob/main/05_dashboard.ipynb)
