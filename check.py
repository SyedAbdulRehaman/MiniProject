prototxt = "models/models_colorization_deploy_v2.prototxt"
model = "models/colorization_release_v2.caffemodel"

# Check file existence
import os
print(os.path.exists(prototxt))  # Check if prototxt file exists
print(os.path.exists(model))     # Check if caffemodel file exists

# Check file permissions (if running into permission issues)
print(os.access(prototxt, os.R_OK))  # Check read access for prototxt
print(os.access(model, os.R_OK))     # Check read access for caffemodel
