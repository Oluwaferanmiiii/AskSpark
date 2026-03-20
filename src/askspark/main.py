"""
Main entry point for AskSpark AI Consultant Assistant
"""

import sys
from .config.logging import setup_logging, get_logger
from .config.settings import Config


def main():
    """Main entry point"""
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        logger.info("Starting AskSpark AI Consultant Assistant")
        
        # Import and start the main application
        from app import main as app_main
        app_main()
        
    except KeyboardInterrupt:
        logger.info("Application stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Application error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
