import audio
import head_pose
import detection
import threading as th
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("Starting head pose thread")
    head_pose_thread = th.Thread(target=head_pose.pose)
    logger.info("Starting audio thread")
    audio_thread = th.Thread(target=audio.sound)
    logger.info("Starting detection thread")
    detection_thread = th.Thread(target=detection.run_detection)

    head_pose_thread.start()
    audio_thread.start()
    detection_thread.start()

    head_pose_thread.join()
    audio_thread.join()
    detection_thread.join()

    logger.info("All threads have finished")
