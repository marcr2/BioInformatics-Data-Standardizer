"""
Centralized Logging Utility for BIDS

Provides structured logging with console and file output.
Supports different log levels and includes timestamps and context.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
import os


class BIDSLogger:
    """
    Centralized logger for BIDS application.
    
    Features:
    - Console output with colored log levels
    - File output for persistent logs
    - Structured format with timestamps
    - Different log levels (DEBUG, INFO, WARNING, ERROR)
    """
    
    # Log level colors for console (ANSI escape codes)
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
    }
    
    def __init__(
        self,
        name: str = "BIDS",
        log_file: Optional[str] = None,
        level: int = logging.DEBUG,
        console_output: bool = True,
        file_output: bool = True
    ):
        """
        Initialize the logger.
        
        Args:
            name: Logger name
            log_file: Path to log file (default: logs/bids.log)
            level: Logging level (default: DEBUG)
            console_output: Whether to output to console
            file_output: Whether to output to file
        """
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        self.logger.handlers = []  # Clear existing handlers
        
        # Create formatters
        console_format = "%(asctime)s [%(levelname)s] %(message)s"
        file_format = "%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
        
        # Console handler with colors
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(ColoredFormatter(console_format))
            self.logger.addHandler(console_handler)
        
        # File handler
        if file_output:
            if log_file is None:
                # Default log file in project root/logs directory
                log_dir = Path(__file__).parent.parent.parent / "logs"
                log_dir.mkdir(exist_ok=True)
                log_file = log_dir / "bids.log"
            else:
                log_file = Path(log_file)
                log_file.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(level)
            file_handler.setFormatter(logging.Formatter(file_format, datefmt='%Y-%m-%d %H:%M:%S'))
            self.logger.addHandler(file_handler)
    
    def debug(self, message: str, context: Optional[str] = None) -> None:
        """Log debug message."""
        self._log(logging.DEBUG, message, context)
    
    def info(self, message: str, context: Optional[str] = None) -> None:
        """Log info message."""
        self._log(logging.INFO, message, context)
    
    def warning(self, message: str, context: Optional[str] = None) -> None:
        """Log warning message."""
        self._log(logging.WARNING, message, context)
    
    def error(self, message: str, context: Optional[str] = None) -> None:
        """Log error message."""
        self._log(logging.ERROR, message, context)
    
    def critical(self, message: str, context: Optional[str] = None) -> None:
        """Log critical message."""
        self._log(logging.CRITICAL, message, context)
    
    def _log(self, level: int, message: str, context: Optional[str] = None) -> None:
        """Internal logging method."""
        if context:
            message = f"[{context}] {message}"
        self.logger.log(level, message)
    
    def log_pipeline_stage(self, stage: str, details: str = "") -> None:
        """Log a pipeline stage change."""
        separator = "-" * 50
        self.info(f"\n{separator}\nPIPELINE STAGE: {stage}\n{details}\n{separator}")
    
    def log_dataframe_info(self, df, name: str = "DataFrame") -> None:
        """Log DataFrame shape and column info."""
        try:
            import pandas as pd
            if isinstance(df, pd.DataFrame):
                self.debug(
                    f"{name}: shape={df.shape}, "
                    f"columns=[{', '.join(df.columns[:5])}{'...' if len(df.columns) > 5 else ''}], "
                    f"memory={df.memory_usage(deep=True).sum() / 1024 / 1024:.2f}MB"
                )
        except Exception as e:
            self.debug(f"{name}: Unable to log DataFrame info - {e}")
    
    def log_execution_time(self, operation: str, start_time: datetime) -> None:
        """Log execution time for an operation."""
        elapsed = (datetime.now() - start_time).total_seconds()
        self.debug(f"{operation} completed in {elapsed:.2f}s")


class ColoredFormatter(logging.Formatter):
    """Formatter that adds colors to log levels for console output."""
    
    COLORS = {
        'DEBUG': '\033[36m',     # Cyan
        'INFO': '\033[32m',      # Green
        'WARNING': '\033[33m',   # Yellow
        'ERROR': '\033[31m',     # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m',      # Reset
    }
    
    def __init__(self, fmt: str):
        super().__init__(fmt, datefmt='%H:%M:%S')
    
    def format(self, record: logging.LogRecord) -> str:
        """Format the log record with colors."""
        # Check if we're in a terminal that supports colors
        if hasattr(sys.stdout, 'isatty') and sys.stdout.isatty():
            levelname = record.levelname
            if levelname in self.COLORS:
                record.levelname = f"{self.COLORS[levelname]}{levelname}{self.COLORS['RESET']}"
        return super().format(record)


# Global logger instance
_logger_instance: Optional[BIDSLogger] = None


def get_logger(
    name: str = "BIDS",
    log_file: Optional[str] = None,
    level: int = logging.DEBUG
) -> BIDSLogger:
    """
    Get or create the global BIDS logger instance.
    
    Args:
        name: Logger name
        log_file: Optional path to log file
        level: Logging level
    
    Returns:
        BIDSLogger instance
    """
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = BIDSLogger(name=name, log_file=log_file, level=level)
    return _logger_instance


def reset_logger() -> None:
    """Reset the global logger instance."""
    global _logger_instance
    _logger_instance = None


# Convenience functions for quick access
def debug(message: str, context: Optional[str] = None) -> None:
    """Log debug message using global logger."""
    get_logger().debug(message, context)


def info(message: str, context: Optional[str] = None) -> None:
    """Log info message using global logger."""
    get_logger().info(message, context)


def warning(message: str, context: Optional[str] = None) -> None:
    """Log warning message using global logger."""
    get_logger().warning(message, context)


def error(message: str, context: Optional[str] = None) -> None:
    """Log error message using global logger."""
    get_logger().error(message, context)


def log_pipeline_stage(stage: str, details: str = "") -> None:
    """Log a pipeline stage change using global logger."""
    get_logger().log_pipeline_stage(stage, details)


def log_dataframe_info(df, name: str = "DataFrame") -> None:
    """Log DataFrame info using global logger."""
    get_logger().log_dataframe_info(df, name)

