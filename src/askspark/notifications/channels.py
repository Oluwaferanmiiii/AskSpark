"""
Notification channel definitions for AskSpark
"""

from enum import Enum
from typing import Dict
from dataclasses import dataclass


class NotificationChannel(Enum):
    """Available notification channels"""
    PUSHOVER = "pushover"
    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"


@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    channel: NotificationChannel
    enabled: bool = True
    config: Dict = None
    
    def __post_init__(self):
        if self.config is None:
            self.config = {}
