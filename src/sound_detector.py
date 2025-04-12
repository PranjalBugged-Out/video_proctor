import logging
import time

logger = logging.getLogger(__name__)

def detect():
    logger.info("Sound detection started")
    try:
        while True:
            # Simulate sound detection logic
            logger.info("Detecting sound...")
            time.sleep(5)  # Simulate time delay for detection
    except KeyboardInterrupt:
        logger.info("Sound detection stopped")

# Ensure the detect function is accessible
if __name__ == "__main__":
    detect()

# Print statement to verify module import
print("sound_detector module imported successfully")
