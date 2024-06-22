import ssl
import urllib.request
import os

ssl._create_default_https_context = ssl._create_unverified_context

# load dataset
from pycaret.datasets import get_data
from pycaret.regression import *

def main():
    # Load dataset
    insurance = get_data('insurance')

    # Init environment
    r1 = setup(insurance, target='charges', session_id=123, normalize=True, polynomial_features=True, bin_numeric_features=['age', 'bmi'])

    # Train a model
    lr = create_model('lr')

    # Determine the base directory path
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Gets the directory of this script

    # Define the relative path within the project folder
    model_dir = os.path.join(base_dir, 'InsureFlow', 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Save pipeline/model
    save_model(lr, model_name=os.path.join(model_dir, 'deployment_12062024'))

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()