#!/bin/bash
# filepath: /Volumes/DATA/DUC/test_script/start.sh

# Download required NLTK data
python -c "
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True) 
nltk.download('wordnet', quiet=True)
print('NLTK data downloaded successfully')
"

# Start the application
exec uvicorn app:app --host 0.0.0.0 --port $PORT