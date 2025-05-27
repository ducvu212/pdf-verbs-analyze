#!/bin/bash
# filepath: /Volumes/DATA/DUC/test_script/start.sh

echo "🚀 Starting PDF Verb Analyzer..."

# Set working directory
cd /opt/render/project/src

# Create directories if they don't exist
mkdir -p uploads exports cache templates
chmod 755 uploads exports cache templates

echo "📁 Created directories:"
ls -la uploads exports cache templates

# Download spaCy model if not exists
echo "📥 Downloading spaCy model..."
python -m spacy download en_core_web_sm --quiet

echo "🔍 Checking Python environment..."
python -c "
import sys
import os
print(f'Python version: {sys.version}')
print(f'Working directory: {os.getcwd()}')
print(f'Python path: {sys.path}')

# Check if directories exist
for dir_name in ['uploads', 'exports', 'cache']:
    exists = os.path.exists(dir_name)
    print(f'{dir_name}: {\"✅\" if exists else \"❌\"} ({\"exists\" if exists else \"missing\"})')
"

echo "🌟 Starting application..."
exec uvicorn app:app --host 0.0.0.0 --port $PORT