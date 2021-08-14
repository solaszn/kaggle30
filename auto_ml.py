# Cloud AutoML
from google.cloud import automl_v1beta1 as automl
automl_client = automl.AutoMlClient()

# Cloud Translation
from google.cloud import translate_v2
translate_client = translate_v2.Client()

# Cloud Natural Language
from google.cloud import language_v1
client = language_v1.LanguageServiceClient()

# Cloud Video Intelligence
from google.cloud import videointelligence
video_client = videointelligence.VideoIntelligenceServiceClient()

# Cloud Vision
from google.cloud import vision
client = vision.ImageAnnotatorClient()

# Cloud Storage
from google.cloud import storage
storage_client = storage.Client(project='YOUR PROJECT ID')


PROJECT_ID = 'YOUR PROJECT ID'
BUCKET_NAME = 'your-bucket-name-here'

# Do not change: Fill in the remaining variables
DATASET_DISPLAY_NAME = 'house_prices_dataset'
TRAIN_FILEPATH = "../input/house-prices-advanced-regression-techniques/train.csv"
TEST_FILEPATH = "../input/house-prices-advanced-regression-techniques/test.csv"
TARGET_COLUMN = 'SalePrice'
ID_COLUMN = 'Id'
MODEL_DISPLAY_NAME = 'house_prices_model'
TRAIN_BUDGET = 2000

# Do not change: Create an instance of the wrapper
from automl_tables_wrapper import AutoMLTablesWrapper

amw = AutoMLTablesWrapper(project_id=PROJECT_ID,
                          bucket_name=BUCKET_NAME,
                          dataset_display_name=DATASET_DISPLAY_NAME,
                          train_filepath=TRAIN_FILEPATH,
                          test_filepath=TEST_FILEPATH,
                          target_column=TARGET_COLUMN,
                          id_column=ID_COLUMN,
                          model_display_name=MODEL_DISPLAY_NAME,
                          train_budget=TRAIN_BUDGET)

# Do not change: Create and train the model
amw.train_model()

# Do not change: Get predictions
amw.get_predictions()

# Note: Google Cloud AutoML Tables is a paid service. At the time of publishing, it charges $19.32 per hour of compute during 
#       training and $1.16 per hour of compute for batch prediction. You can find more details here.