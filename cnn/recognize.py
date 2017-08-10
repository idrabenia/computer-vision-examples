import os
import sys
sys.path.append(os.path.abspath('../..'))
import classify_image as ci


ci.run_inference_on_images('./Data/Leopards', './model')
ci.run_inference_on_images('./Data/crocodile', './model')
