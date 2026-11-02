"""Google Calendar availability and bookings."""

import datetime
from zoneinfo import ZoneInfo

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain.tools import tool
from pydantic import BaseModel, Field

from tools.common import CREDENTIALS_PATH, SECRETS_DIR, TOKEN_PATH, logger


def get_calendar_service():
    """
    Handles OAuth2 authentication and returns the Google Calendar service.

    Credentials are expected in:
    - secrets/credentials.json
    - secrets/token.json
    """

    SCOPES = ["https://www.googleapis.com/auth/calendar"]
    creds = None

    if TOKEN_PATH.exists():
        creds = Credentials.from_authorized_user_file(str(TOKEN_PATH), SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            if not CREDENTIALS_PATH.exists():
                raise FileNotFoundError(f"Google credentials file not found: {CREDENTIALS_PATH}")

            flow = InstalledAppFlow.from_client_secrets_file(
                str(CREDENTIALS_PATH),
                SCOPES,
            )
            creds = flow.run_local_server(port=0)

        SECRETS_DIR.mkdir(parents=True, exist_ok=True)
        with open(TOKEN_PATH, "w", encoding="utf-8") as token_file:
            token_file.write(creds.to_json())

    return build("calendar", "v3", credentials=creds)


CALENDAR_TIMEZONE = "Europe/Rome"
APPOINTMENT_DURATION_HOURS = 1


class AvailabilityInput(BaseModel):
    date_str: str = Field(description="Date to check in YYYY-MM-DD format.")
    time_str: str | None = Field(
        default=None,
        description=(
            "Optional specific time in HH:MM (24-hour) to check. "
            "If omitted, returns all busy slots for the whole day."
        ),
    )
    duration_hours: float = Field(
        default=APPOINTMENT_DURATION_HOURS,
        description="Expected duration of the appointment in hours, used to detect slot conflicts.",
    )


@tool(args_schema=AvailabilityInput)
def check_calendar_availability_tool(
    date_str: str, time_str: str | None = None, duration_hours: float = APPOINTMENT_DURATION_HOURS
) -> str:
    """
    Checks the installer's Google Calendar for free/busy slots.

    Use this when the user asks:
    - 'When are you free on <date>?'
    - 'Is <date> at <time> available?'
    - Before confirming a booking to prevent double-booking.

    Returns a plain-English summary of busy periods and whether the
    requested time slot is free.
    """
    try:
        timezone = ZoneInfo(CALENDAR_TIMEZONE)
        target_date = datetime.datetime.strptime(date_str, "%Y-%m-%d").date()
        day_start = datetime.datetime.combine(target_date, datetime.time.min).replace(
            tzinfo=timezone
        )
        day_end = datetime.datetime.combine(target_date, datetime.time.max).replace(tzinfo=timezone)

        service = get_calendar_service()
        body = {
            "timeMin": day_start.isoformat(),
            "timeMax": day_end.isoformat(),
            "timeZone": CALENDAR_TIMEZONE,
            "items": [{"id": "primary"}],
        }
        result = service.freebusy().query(body=body).execute()
        busy_slots = result.get("calendars", {}).get("primary", {}).get("busy", [])

        if not busy_slots:
            base_msg = f"The calendar is completely free on {date_str}."
        else:
            formatted = []
            for slot in busy_slots:
                start = datetime.datetime.fromisoformat(slot["start"]).astimezone(timezone)
                end = datetime.datetime.fromisoformat(slot["end"]).astimezone(timezone)
                formatted.append(f"{start.strftime('%H:%M')}–{end.strftime('%H:%M')}")
            base_msg = f"Busy periods on {date_str}: {', '.join(formatted)}."

        if not time_str:
            return base_msg

        # Check whether the specific requested slot is free.
        requested_start = datetime.datetime.strptime(
            f"{date_str} {time_str}", "%Y-%m-%d %H:%M"
        ).replace(tzinfo=timezone)
        requested_end = requested_start + datetime.timedelta(hours=duration_hours)

        for slot in busy_slots:
            busy_start = datetime.datetime.fromisoformat(slot["start"]).astimezone(timezone)
            busy_end = datetime.datetime.fromisoformat(slot["end"]).astimezone(timezone)
            if requested_start < busy_end and requested_end > busy_start:
                return (
                    f"{base_msg} "
                    f"The slot {time_str}–{requested_end.strftime('%H:%M')} is NOT available."
                )

        return f"{base_msg} The slot {time_str}–{requested_end.strftime('%H:%M')} is available."

    except ValueError as exc:
        return f"Availability Error: invalid date/time format — {exc}"
    except FileNotFoundError as exc:
        logger.error(f"AVAILABILITY_CREDENTIALS_MISSING: {exc}")
        return "Error: Google Calendar credentials are missing. Check the secrets/ folder."
    except Exception as exc:
        logger.error(f"AVAILABILITY_FAILURE: {exc}", exc_info=True)
        return "Error: Could not retrieve calendar availability."


class BookingInput(BaseModel):
    date_str: str = Field(description="Date in YYYY-MM-DD format.")
    time_str: str = Field(description="Time in HH:MM format, 24-hour clock.")
    model: str = Field(description="Specific hardware model code.")
    purpose: str = Field(
        default="Installation",
        description="Short label for the calendar event, e.g. 'Installation' or 'Survey'.",
    )
    duration_hours: float = Field(
        default=APPOINTMENT_DURATION_HOURS,
        description="Appointment duration in hours (e.g. quantity × hours-per-unit + round-trip travel time).",
    )


@tool(args_schema=BookingInput)
def book_installation_tool(
    date_str: str,
    time_str: str,
    model: str,
    purpose: str = "Installation",
    duration_hours: float = APPOINTMENT_DURATION_HOURS,
) -> str:
    """
    Schedules a professional appointment (installation or site survey) for a
    specific hardware model. Registers the event in Google Calendar.
    """

    model = model.upper().strip()

    try:
        booking_dt = datetime.datetime.strptime(
            f"{date_str} {time_str}",
            "%Y-%m-%d %H:%M",
        )

        timezone_name = "Europe/Rome"
        timezone = ZoneInfo(timezone_name)
        booking_dt_aware = booking_dt.replace(tzinfo=timezone)
        now_aware = datetime.datetime.now(timezone)

        if booking_dt_aware < now_aware:
            logger.warning(f"BOOKING_REJECTED: Past date attempted ({date_str})")
            return f"Error: Cannot book for {date_str}. Please choose a future date."

        # Guard: reject the booking if the slot is already taken.
        availability = check_calendar_availability_tool.invoke(
            {"date_str": date_str, "time_str": time_str, "duration_hours": duration_hours}
        )
        if "NOT available" in availability:
            return (
                f"Booking rejected: the slot {time_str} on {date_str} is already taken. "
                f"Details: {availability}"
            )

        service = get_calendar_service()

        event = {
            "summary": f"{purpose}: {model}",
            "description": f"{purpose} of hardware model {model}.",
            "start": {
                "dateTime": booking_dt_aware.isoformat(),
                "timeZone": timezone_name,
            },
            "end": {
                "dateTime": (
                    booking_dt_aware + datetime.timedelta(hours=duration_hours)
                ).isoformat(),
                "timeZone": timezone_name,
            },
        }

        service.events().insert(calendarId="primary", body=event).execute()

        confirmation = (
            f"CONFIRMED: {purpose} for {model} scheduled for "
            f"{booking_dt.strftime('%A, %B %d, %Y')} at {time_str} "
            "and synced to Google Calendar."
        )

        logger.info(
            f"BOOKING_SUCCESS: {purpose} {model} for {date_str} {time_str} ({duration_hours}h)"
        )
        return confirmation

    except ValueError:
        return "Error: Invalid format. Use YYYY-MM-DD for date and HH:MM for time."

    except FileNotFoundError as exc:
        logger.error(f"BOOKING_CREDENTIALS_MISSING: {str(exc)}")
        return "Error: Google Calendar credentials are missing. Check the secrets/ folder."

    except Exception as exc:
        logger.error(f"BOOKING_CRITICAL_FAILURE: {str(exc)}", exc_info=True)
        return "Error: The booking system encountered an internal failure."
