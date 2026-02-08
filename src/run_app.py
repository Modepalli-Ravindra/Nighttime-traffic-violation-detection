import os
import sys
from app.app import create_app

def create_app_with_config():
    """Create the Flask app with proper configuration for the modular structure."""
    # Get the directory containing this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths relative to the project root
    template_dir = os.path.join(base_dir, 'ui', 'templates')
    static_dir = os.path.join(base_dir, 'ui', 'static')
    
    # Create Flask app with correct template and static directories
    app = create_app(template_dir, static_dir)
    
    return app

if __name__ == '__main__':
    app = create_app_with_config()
    app.run(debug=True, host='0.0.0.0', port=5000)