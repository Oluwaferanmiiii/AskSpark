"""
Multi-channel notification service for AskSpark
"""

import logging
import requests
import smtplib
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

try:
    from pushover import Client
except ImportError:
    Client = None

from .channels import NotificationChannel
from ..config.logging import get_logger

logger = get_logger(__name__)


class NotificationService:
    """Multi-channel notification service"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.pushover_client = None
        
        # Initialize Pushover if configured
        if (Client and 
            config.get('PUSHOVER_USER_KEY') and 
            config.get('PUSHOVER_APP_TOKEN')):
            try:
                self.pushover_client = Client(
                    config['PUSHOVER_USER_KEY'],
                    api_token=config['PUSHOVER_APP_TOKEN']
                )
                logger.info("Pushover client initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Pushover client: {e}")
    
    def send_notification(self, 
                         message: str, 
                         title: str = "AI Consultant Alert",
                         channels: List[NotificationChannel] = None,
                         channel_config: Dict = None) -> bool:
        """
        Send notification through multiple channels
        
        Args:
            message: Notification message
            title: Notification title
            channels: List of channels to use
            channel_config: Channel-specific configuration
            
        Returns:
            True if at least one channel succeeded
        """
        if channels is None:
            channels = [NotificationChannel.PUSHOVER]
        
        if channel_config is None:
            channel_config = {}
        
        success = False
        
        for channel in channels:
            try:
                if channel == NotificationChannel.PUSHOVER:
                    success |= self._send_pushover(message, title, channel_config.get('pushover', {}))
                elif channel == NotificationChannel.EMAIL:
                    success |= self._send_email(message, title, channel_config.get('email', {}))
                elif channel == NotificationChannel.SLACK:
                    success |= self._send_slack(message, title, channel_config.get('slack', {}))
                elif channel == NotificationChannel.WEBHOOK:
                    success |= self._send_webhook(message, title, channel_config.get('webhook', {}))
            except Exception as e:
                logger.error(f"Failed to send notification via {channel.value}: {e}")
        
        return success
    
    def _send_pushover(self, message: str, title: str, config: Dict) -> bool:
        """Send Pushover notification"""
        if not self.pushover_client:
            logger.warning("Pushover not configured")
            return False
        
        try:
            self.pushover_client.send_message(message, title=title)
            logger.info("Pushover notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Pushover notification failed: {e}")
            return False
    
    def _send_email(self, message: str, title: str, config: Dict) -> bool:
        """Send email notification"""
        required_fields = ['smtp_server', 'smtp_port', 'sender_email', 'sender_password', 'recipient']
        
        for field in required_fields:
            if field not in config:
                logger.warning(f"Email configuration missing {field}")
                return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = config['sender_email']
            msg['To'] = config['recipient']
            msg['Subject'] = title
            
            msg.attach(MIMEText(message, 'plain'))
            
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['sender_email'], config['sender_password'])
            server.send_message(msg)
            server.quit()
            
            logger.info("Email notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Email notification failed: {e}")
            return False
    
    def _send_slack(self, message: str, title: str, config: Dict) -> bool:
        """Send Slack notification"""
        if 'webhook_url' not in config:
            logger.warning("Slack webhook URL not configured")
            return False
        
        try:
            payload = {
                "text": f"*{title}*\n{message}",
                "username": "AI Consultant Assistant"
            }
            
            response = requests.post(config['webhook_url'], json=payload, timeout=10)
            response.raise_for_status()
            
            logger.info("Slack notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Slack notification failed: {e}")
            return False
    
    def _send_webhook(self, message: str, title: str, config: Dict) -> bool:
        """Send generic webhook notification"""
        if 'url' not in config:
            logger.warning("Webhook URL not configured")
            return False
        
        try:
            payload = {
                "title": title,
                "message": message,
                "timestamp": datetime.now().isoformat(),
                "source": "AI Consultant Assistant"
            }
            
            # Add custom headers if provided
            headers = config.get('headers', {})
            headers['Content-Type'] = 'application/json'
            
            response = requests.post(config['url'], json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            
            logger.info("Webhook notification sent successfully")
            return True
        except Exception as e:
            logger.error(f"Webhook notification failed: {e}")
            return False
