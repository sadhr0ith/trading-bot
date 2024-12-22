import smtplib
from email.mime.text import MIMEText 
from  email.mime.multipart import MIMEMultipart
import os
from utils.logger import setup_logger

logger = setup_logger("Email Notification")

app_password = os.getenv('GMAIL_APP_PASSWORD')

# Use the app password instead of your Gmail password
def send_email(subject, body, to_email):
    """
    Send an email notification with the specified subject and body.
    :param subject: Email subject
    :param body: Email body text
    :param to_email: Recipient email address
    """
    # try:
        # Check for None values in subject or body
    if subject is None or body is None:
        raise ValueError("Subject or body is None. Cannot send email.")

    if isinstance(to_email, list):
            to_email = ', '.join(to_email)

    # Gmail SMTP settings
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    from_email = "sadhroith@gmail.com"
    from_password = app_password

    # Create the email
    msg = MIMEMultipart()
    msg['From'] = from_email
    msg['To'] = to_email
    msg['Subject'] = subject

    # Attach the email body
    msg.attach(MIMEText(body, 'plain'))

    # Set up the server
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()  # Start TLS encryption
    server.login(from_email, from_password)

    # Send the email
    server.sendmail(from_email, to_email, msg.as_string())
    server.quit()

    logger.info(f"Email notification sent to {to_email}.")
    # except Exception as e:
    #     logger.error(f"Failed to send email: {str(e)}")
