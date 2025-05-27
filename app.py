import os
import sys
import logging
from pathlib import Path

# Configure logging first
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, File, UploadFile, Request, HTTPException, BackgroundTasks
    from fastapi.responses import HTMLResponse, FileResponse
    from fastapi.staticfiles import StaticFiles
    from fastapi.templating import Jinja2Templates
    from fastapi.middleware.cors import CORSMiddleware
    logger.info("✅ FastAPI imports successful")
except ImportError as e:
    logger.error(f"❌ FastAPI import failed: {e}")
    sys.exit(1)

# Production configuration
IS_PRODUCTION = os.getenv("RENDER") is not None or os.getenv("RAILWAY_ENVIRONMENT") is not None
PORT = int(os.environ.get("PORT", 8000))

logger.info(f"🔧 Configuration: IS_PRODUCTION={IS_PRODUCTION}, PORT={PORT}")

# Configure paths
if IS_PRODUCTION:
    BASE_DIR = Path("/opt/render/project/src")
else:
    BASE_DIR = Path(__file__).parent

logger.info(f"📁 BASE_DIR: {BASE_DIR}")

UPLOAD_FOLDER = BASE_DIR / "uploads"
EXPORT_FOLDER = BASE_DIR / "exports" 
CACHE_FOLDER = BASE_DIR / "cache"
TEMPLATES_FOLDER = BASE_DIR / "templates"
STATIC_FOLDER = BASE_DIR / "static"

# Ensure directories exist với proper permissions
def ensure_directories():
    """Ensure all required directories exist with proper permissions"""
    directories = [UPLOAD_FOLDER, EXPORT_FOLDER, CACHE_FOLDER, TEMPLATES_FOLDER, STATIC_FOLDER]
    
    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
            os.chmod(directory, 0o755)
            logger.info(f"✅ Directory ensured: {directory}")
            
            # Create .gitkeep for static
            if directory == STATIC_FOLDER:
                gitkeep = directory / ".gitkeep"
                if not gitkeep.exists():
                    gitkeep.write_text("")
                    
        except Exception as e:
            logger.error(f"❌ Failed to create directory {directory}: {e}")
            if IS_PRODUCTION:
                # Use /tmp as fallback
                fallback_dir = Path("/tmp") / f"pdf_analyzer_{directory.name}"
                fallback_dir.mkdir(parents=True, exist_ok=True)
                logger.warning(f"Using fallback: {fallback_dir}")

# Initialize directories
ensure_directories()

# Initialize FastAPI app
app = FastAPI(
    title="PDF Verb Analyzer",
    description="Analyze verb tenses in PDF documents",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files with error handling
try:
    if STATIC_FOLDER.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_FOLDER)), name="static")
        logger.info(f"✅ Static files mounted: {STATIC_FOLDER}")
    else:
        logger.warning(f"⚠️ Static directory not found: {STATIC_FOLDER}")
except Exception as e:
    logger.error(f"❌ Failed to mount static files: {e}")

# Initialize templates
try:
    templates = Jinja2Templates(directory=str(TEMPLATES_FOLDER))
    logger.info(f"✅ Templates initialized: {TEMPLATES_FOLDER}")
except Exception as e:
    logger.error(f"❌ Templates initialization failed: {e}")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring"""
    try:
        # Check directories
        dirs_status = {}
        for name, path in [
            ("uploads", UPLOAD_FOLDER),
            ("exports", EXPORT_FOLDER), 
            ("cache", CACHE_FOLDER),
            ("templates", TEMPLATES_FOLDER),
            ("static", STATIC_FOLDER)
        ]:
            dirs_status[name] = path.exists()
        
        return {
            "status": "healthy",
            "message": "PDF Verb Analyzer is running",
            "version": "1.0.0",
            "port": PORT,
            "is_production": IS_PRODUCTION,
            "directories": dirs_status,
            "working_directory": str(Path.cwd())
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "message": f"Health check failed: {str(e)}"
        }

# Root endpoint
@app.get("/")
async def read_root(request: Request):
    """Root endpoint"""
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except Exception as e:
        logger.error(f"Root endpoint failed: {e}")
        return HTMLResponse(content=f"<h1>PDF Verb Analyzer</h1><p>Service is running but templates not found: {e}</p>")

# Test endpoint
@app.get("/test")
async def test_endpoint():
    """Simple test endpoint"""
    return {
        "message": "PDF Verb Analyzer is working!",
        "port": PORT,
        "working_directory": str(Path.cwd()),
        "base_directory": str(BASE_DIR)
    }

# ... rest of your existing endpoints ...

logger.info("🚀 PDF Verb Analyzer initialized successfully")

# For running with uvicorn directly
if __name__ == "__main__":
    import uvicorn
    logger.info(f"🌟 Starting server on port {PORT}")
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info",
        access_log=True
    )