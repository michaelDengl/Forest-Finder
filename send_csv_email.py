import os
import smtplib
from email.message import EmailMessage

# === CONFIGURATION ===
CSV_FOLDER = "/home/lubuharg/Documents/MyScanner/MTG/Output"
EMAIL_FROM = "magiccardsender@gmail.com"
EMAIL_TO = "magictg.verein@gmail.com"
EMAIL_SUBJECT = "MTG CSV Report"
EMAIL_BODY = "Hi, \nAttached is the latest scanned Magic card report. \nGreetings\nLego Scanner"

# !!! Replace with your real credentials or use a secure method to load them
GMAIL_USER = "magiccardsender@gmail.com"
GMAIL_APP_PASSWORD = 'wbzy nuai khai kbte'  # NOT your regular Gmail password

def send_csv_files():
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    msg["Subject"] = EMAIL_SUBJECT
    msg.set_content(EMAIL_BODY)

    # Attach all .csv files in the output folder
    for file in os.listdir(CSV_FOLDER):
        if file.endswith(".csv"):
            file_path = os.path.join(CSV_FOLDER, file)
            with open(file_path, "rb") as f:
                data = f.read()
                msg.add_attachment(data, maintype="text", subtype="csv", filename=file)

    # Send email via Gmail SMTP
    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(GMAIL_USER, GMAIL_APP_PASSWORD)
            smtp.send_message(msg)
        print(f"[✓] Email sent to {EMAIL_TO}")
    except Exception as e:
        print(f"[ERROR] Failed to send email: {e}")

if __name__ == "__main__":
    send_csv_files()
