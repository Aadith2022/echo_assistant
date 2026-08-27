import os
import datetime
from google.auth.exceptions import RefreshError
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
import config

SCOPES = ["https://www.googleapis.com/auth/calendar.readonly"]
TOKEN_PATH = os.path.join(config.BASE_DIR, "token.json")


def _get_service():
    if not os.path.exists(config.GOOGLE_CREDENTIALS_PATH):
        return None

    creds = None
    if os.path.exists(TOKEN_PATH):
        creds = Credentials.from_authorized_user_file(TOKEN_PATH, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except RefreshError:
                # Refresh tokens for an app still in "Testing" publishing status
                # expire after 7 days. Re-consent needs a browser and a human;
                # doing that from inside a tool call would block the turn on a
                # window the user may not even be looking at.
                return None
        else:
            flow = InstalledAppFlow.from_client_secrets_file(config.GOOGLE_CREDENTIALS_PATH, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_PATH, "w") as f:
            f.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


def get_upcoming_events(max_results: int = 5) -> str:
    """List the user's upcoming Google Calendar events."""
    service = _get_service()
    if service is None:
        return (
            "Calendar is unavailable - either no google_credentials.json was found or "
            "the saved authorisation has expired and needs re-granting. Continue without it."
        )

    now = datetime.datetime.utcnow().isoformat() + "Z"
    events_result = service.events().list(
        calendarId="primary",
        timeMin=now,
        maxResults=max_results,
        singleEvents=True,
        orderBy="startTime",
    ).execute()
    events = events_result.get("items", [])

    if not events:
        return "No upcoming events found."

    lines = []
    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        lines.append(f"- {start}: {event.get('summary', '(no title)')}")
    return "\n".join(lines)
