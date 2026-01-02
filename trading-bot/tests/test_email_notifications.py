import os
from unittest.mock import patch

from trading_bot.utils.email_notifications import send_email


def test_send_email_missing_gmail_sender():
    """Test 5.5: send_email without GMAIL_SENDER_EMAIL should return False with error"""
    with patch.dict(os.environ, {}, clear=True):
        result = send_email("Test Subject", "Test Body", "recipient@example.com")
        assert result is False


def test_send_email_missing_gmail_password():
    """Test 5.5: send_email without GMAIL_APP_PASSWORD should return False with error"""
    with patch.dict(os.environ, {"GMAIL_SENDER_EMAIL": "sender@gmail.com"}, clear=True):
        result = send_email("Test Subject", "Test Body", "recipient@example.com")
        assert result is False


def test_send_email_missing_subject():
    """Test: send_email with missing subject should return False"""
    with patch.dict(
        os.environ, {"GMAIL_SENDER_EMAIL": "sender@gmail.com", "GMAIL_APP_PASSWORD": "password"}, clear=True
    ):
        result = send_email("", "Test Body", "recipient@example.com")
        assert result is False


def test_send_email_missing_body():
    """Test: send_email with missing body should return False"""
    with patch.dict(
        os.environ, {"GMAIL_SENDER_EMAIL": "sender@gmail.com", "GMAIL_APP_PASSWORD": "password"}, clear=True
    ):
        result = send_email("Test Subject", "", "recipient@example.com")
        assert result is False


def test_send_email_empty_recipients():
    """Test: send_email with empty recipients should return False"""
    with patch.dict(
        os.environ, {"GMAIL_SENDER_EMAIL": "sender@gmail.com", "GMAIL_APP_PASSWORD": "password"}, clear=True
    ):
        result = send_email("Test Subject", "Test Body", "")
        assert result is False
