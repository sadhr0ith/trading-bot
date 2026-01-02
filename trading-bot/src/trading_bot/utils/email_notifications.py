import smtplib
from collections.abc import Iterable
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from trading_bot.models.env_settings import load_email_settings
from trading_bot.utils.logger import setup_logger

logger = setup_logger("Email Notification")


def _coerce_recipients(recipients: str | Iterable[str]) -> list[str]:
    if isinstance(recipients, str):
        recipients = [recipients]
    elif recipients is None:
        recipients = []
    else:
        recipients = list(recipients)
    return [r for r in recipients if r]


def send_email(subject: str, body: str, to_email: str | Iterable[str]) -> bool:
    """
    Send an email notification with the specified subject and body.
    Returns True on success, False otherwise.
    """
    if not subject or not body:
        logger.warning("Skipping email notification: subject or body is missing.")
        return False

    recipients = _coerce_recipients(to_email)
    if not recipients:
        logger.info("Skipping email notification: recipient list is empty.")
        return False

    email_settings = load_email_settings(logger)
    if not email_settings:
        return False

    from_email = email_settings.sender_email
    app_password = email_settings.app_password

    smtp_server = email_settings.smtp_server
    smtp_port = email_settings.smtp_port

    msg = MIMEMultipart()
    msg["From"] = from_email
    msg["To"] = ", ".join(recipients)
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP(smtp_server, smtp_port, timeout=10)
        server.starttls()
        server.login(from_email, app_password)
        server.sendmail(from_email, recipients, msg.as_string())
        server.quit()
        logger.info(f"Email notification sent to {recipients}.")
        return True
    except (smtplib.SMTPException, OSError, ConnectionError, TimeoutError) as exc:
        logger.error(f"Failed to send email: {exc}")
        return False
