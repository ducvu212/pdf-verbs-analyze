#!/bin/bash
# filepath: /Volumes/DATA/DUC/test_script/start.sh

echo "🚀 Starting PDF Verb Analyzer..."

# Set working directory
cd /opt/render/project/src

# Create directories if they don't exist (including static)
mkdir -p uploads exports cache templates static
chmod 755 uploads exports cache templates static

echo "📁 Created directories:"
ls -la uploads exports cache templates static

# Create .gitkeep for static directory
touch static/.gitkeep

# Download spaCy model if not exists
echo "📥 Downloading spaCy model..."
python -m spacy download en_core_web_sm --quiet

# Verify spaCy installation
echo "🔍 Verifying spaCy model..."
python -c "
import spacy
try:
    nlp = spacy.load('en_core_web_sm')
    print('✅ spaCy model loaded successfully')
except Exception as e:
    print(f'❌ spaCy model loading failed: {e}')
    exit(1)
"

echo "🔍 Checking Python environment..."
python -c "
import sys
import os
print(f'Python version: {sys.version}')
print(f'Working directory: {os.getcwd()}')
print(f'Python path: {sys.path}')
print(f'PORT environment variable: {os.environ.get(\"PORT\", \"Not set\")}')

# Check if directories exist
for dir_name in ['uploads', 'exports', 'cache', 'templates', 'static']:
    exists = os.path.exists(dir_name)
    print(f'{dir_name}: {\"✅\" if exists else \"❌\"} ({\"exists\" if exists else \"missing\"})')
"

# Test import of main app
echo "🧪 Testing app import..."
python -c "
try:
    from app import app
    print('✅ App imported successfully')
except Exception as e:
    print(f'❌ App import failed: {e}')
    import traceback
    traceback.print_exc()
    exit(1)
"

# Set default port if not provided
export PORT=${PORT:-8000}

echo "🌟 Starting application on port $PORT..."
echo "🔗 App will be available at: http://0.0.0.0:$PORT"

# Start with explicit port binding and logging
exec uvicorn app:app \
    --host 0.0.0.0 \
    --port $PORT \
    --log-level info \
    --access-log \
    --no-use-colors