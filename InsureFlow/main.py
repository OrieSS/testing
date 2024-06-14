import ssl
import urllib.request

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

    # Save pipeline/model
    save_model(lr, model_name='C:/Git/AIProject/testing/InsureFlow/deployment_12062024')

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()