import os
import re
import smtplib
from email.message import EmailMessage


def recipients() -> list[str]:
    """The mailing list, from EMAIL_RECIPIENTS (.env locally, a secret on GitHub): the repo is
    public, so addresses never go in config.yaml. Separate with commas, spaces or newlines."""
    found = [a for a in re.split(r"[,;\s]+", os.environ.get("EMAIL_RECIPIENTS", "")) if a]
    if not found:
        raise SystemExit("EMAIL_RECIPIENTS is empty: set it in .env (or as a GitHub Actions secret)")
    return found


def send(subject: str, text: str, html: str) -> int:
    """One message per recipient, so addresses on the list are never shown to each other.
    Returns how many were sent (addresses are not printed: Action logs are public)."""
    to_all = recipients()
    user = os.environ["SMTP_USER"]
    with smtplib.SMTP_SSL(os.environ.get("SMTP_HOST", "smtp.gmail.com"),
                          int(os.environ.get("SMTP_PORT", 465))) as s:
        s.login(user, os.environ["SMTP_PASS"])
        for to in to_all:
            msg = EmailMessage()
            msg["Subject"], msg["From"], msg["To"] = subject, user, to
            msg.set_content(text)
            msg.add_alternative(html, subtype="html")
            s.send_message(msg)
    return len(to_all)
