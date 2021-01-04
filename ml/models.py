import joblib
import os

def save_model(pipe, model_folder):
    
    if not os.path.isdir(model_folder):
        os.mkdir(model_folder)
    joblib.dump(pipe, model_folder + "/pipe.joblib", compress=True)