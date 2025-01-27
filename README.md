# Insurance Charges Prediction App

This Streamlit app predicts insurance charges based on user inputs such as age, sex, BMI, number of children, smoking status, and region. The app also displays the model's performance metrics, including Mean Squared Error (MSE) and R² score.

## Features

- Interactive user input via a sidebar.
- Predict insurance charges using a pre-trained Linear Regression model.
- Display performance metrics of the model (MSE and R² score).
- Clean and user-friendly interface.

## Installation

1. Clone this repository:
   ```bash
   git clone <repository_url>
   cd <repository_folder>
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   source venv/bin/activate  # For Linux/Mac
   venv\Scripts\activate   # For Windows
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   streamlit run streamlit_regression_app.py
   ```

## Usage

- Adjust the input parameters in the sidebar.
- Click the **Predict Charges** button to get the predicted insurance charges.
- View the model's performance metrics at the bottom of the page.

## Files

- `streamlit_regression_app.py`: The main application file.
- `ML.ipynb`: The best Jupyter Notebook file in this repository, showcasing the complete machine learning pipeline, from data preprocessing to model evaluation, with detailed explanations and visualizations.
- `other_notebook.ipynb`: An additional notebook for reference, but `ML.ipynb` is more comprehensive and recommended.
- `linear_regression_model.pkl`: The pre-trained model file.
- `requirements.txt`: List of required Python libraries.

## Requirements

Refer to `requirements.txt` for the full list of dependencies.

## Demo

![Demo Screenshot](screenshot.png)  
*A screenshot of the app interface.*

## Author

Developed by **Bilal Rafique**.

---

## License

This project is licensed under the MIT License.

