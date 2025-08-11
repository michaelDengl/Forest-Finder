import os
import sys
import smtplib
import mimetypes
from email.message import EmailMessage

# === CONFIGURATION ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FOLDER = os.path.join(BASE_DIR, "Output")
EMAIL_FROM = "magiccardsender@gmail.com"
EMAIL_TO = "magictg.verein@gmail.com"
EMAIL_SUBJECT = "MTG CSV Report"
EMAIL_BODY = "Hi,\nAttached is the latest scanned Magic card report.\n\nGreetings\nLego Scanner"

# !!! Replace with your real credentials or use a secure method to load them
GMAIL_USER = "magiccardsender@gmail.com"
GMAIL_APP_PASSWORD = "wbzy nuai khai kbte"  # App password

def _attach_file(msg: EmailMessage, path: str, fallback_name: str | None = None):
    """
    Attach a file to the EmailMessage with a guessed MIME type.
    """
    if not os.path.isfile(path):
        print(f"[WARN] Attachment not found: {path}")
        return

    ctype, encoding = mimetypes.guess_type(path)
    if ctype is None or encoding is not None:
        ctype = "application/octet-stream"
    maintype, subtype = ctype.split("/", 1)

    filename = os.path.basename(path) if fallback_name is None else fallback_name

    with open(path, "rb") as f:
        data = f.read()
    msg.add_attachment(data, maintype=maintype, subtype=subtype, filename=filename)

def send_csv_email(csv_path: str, attachments: list[str] | None = None):
    """
    Send a single CSV with optional image/file attachments.
    """
    attachments = attachments or []

    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = EMAIL_SUBJECT
    msg.set_content(EMAIL_BODY)

    # Attach the CSV
    _attach_file(msg, csv_path)

    # Attach any additional files (e.g., NOT FOUND OCR debug strips)
    for apath in attachments:
        _attach_file(msg, apath)

    # Send email via Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"[✓] Email sent to {EMAIL_TO} with CSV '{os.path.basename(csv_path)}' and {len(attachments)} attachment(s).")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

def send_all_csv_in_folder():
    """
    Legacy fallback: send all CSVs found in CSV_FOLDER.
    """
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = EMAIL_SUBJECT
    msg.set_content(EMAIL_BODY)

    count = 0
    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):
            file_path = os.path.join(CSV_FOLDER, file)
            _attach_file(msg, file_path)
            count += 1

    if count == 0:
        print(f"[WARN] No CSV files found in {CSV_FOLDER}. Sending email without CSVs.")

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"[✓] Email sent to {EMAIL_TO} with {count} CSV file(s).")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

if __name__ == "__main__":
    # Usage:
    #   python3 send_csv_email.py <csv_path> <attachment1> <attachment2> ...
    # or without args:
    #   python3 send_csv_email.py
    if len(sys.argv) >= 2:
        csv_path = sys.argv[1]
        other_attachments = sys.argv[2:]
        send_csv_email(csv_path, other_attachments)
    else:
        send_all_csv_in_folder()
