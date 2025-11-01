from loguru import logger
from datetime import datetime
from pathlib import Path
import sys
import time


class LoggedProcess:
    """
    Class which automatically configures the logger.
    Logs errors to file and all other messages to console with color and alignment.
    """

    def __init__(self, output_dir: str = "logs"):
        self.output_dir = Path(output_dir)
        self.make_output_directory()
        self.setup_logger()

    @logger.catch(message="Failed to create output directory", reraise=True)
    def make_output_directory(self):
        """Create required output directories"""
        for sub in ["general", "error"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    @logger.catch(message="Logging setup failed", reraise=True)
    def setup_logger(self):
        """
        Set up loguru handlers for console and error files.
        """
        # Remove any default handlers
        logger.remove()
        timestamp = datetime.now().strftime("%d_%H%M")
        error_dir = self.output_dir / "error"

        # for CLI
        logger.add(
            sys.stderr,
            level="INFO",
        )

        # errors written to disk
        logger.add(
            error_dir / f"error_{timestamp}.log",
            format="{time:D MMMM, YYYY > HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
            level="ERROR",
            backtrace=True,
            diagnose=True,
        )

        # log all events
        logger.add(
            self.output_dir / "general" / f"all_{timestamp}.log",
            format="{time:D MMMM, YYYY > HH:mm:ss} | {level} | {message}",
            level="DEBUG",
            # rotation="10 MB",
            retention="7 days",
        )

        logger.success(f"Logger initialized — logs will be saved to: {self.output_dir}")

        return logger


if __name__ == "__main__":
    log_process = LoggedProcess(output_dir="./test_logs")

    # Test different log levels
    logger.debug("This is a debug message")
    time.sleep(1)
    logger.info("This is an info message")
    time.sleep(1)
    logger.warning("This is a warning message")
    time.sleep(1)
    logger.error("This is an error message")
    time.sleep(1)
    logger.success("This is a success message")
    time.sleep(1)

    try:
        result = 1 / 0
    except Exception as e:
        logger.error(f"Division by zero error: {e}")
        logger.exception("Exception with full traceback:")
