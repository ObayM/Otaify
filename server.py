"""
FastAPI server to search recent emails from a Gmail account via the Gmail API,
based on sender, recipient, and time frame. Extracts relevant information,
detects forwarded emails, saves matching emails as styled HTML files,
and returns the paths to these files.

**Requires `token.json` to exist (run script once locally first for auth).**
"""

import os
import base64
import json
import re
import argparse
import logging
import asyncio
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple


# FastAPI & Related
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, Depends
from pydantic import BaseModel, Field, EmailStr
import uvicorn
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware


# Google API Libraries
from googleapiclient.discovery import build, Resource
from googleapiclient.errors import HttpError
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import email.utils

# --- Configuration ---

# Scopes required for the Gmail API.
SCOPES: List[str] = ['https://www.googleapis.com/auth/gmail.readonly']

# Default path for storing API credentials token.
DEFAULT_TOKEN_PATH: str = 'token.json'
# Default path for the client secrets file obtained from Google Cloud Console.
DEFAULT_CREDENTIALS_PATH: str = 'credentials.json'
# Default directory to save extracted email HTML files.
DEFAULT_OUTPUT_DIR: str = './temp/extracted_mails'
# Preferred MIME types for email body content, in order of preference.
PREFERRED_MIME_TYPES: Tuple[str, str] = ('text/html', 'text/plain')
# Strings indicating a forwarded message within the content.
FORWARDED_INDICATORS: Tuple[str, str] = (
    '---------- Forwarded message ---------',
    '-------- Original Message --------'
)
# Subject prefixes indicating a forwarded message.
FORWARDED_SUBJECT_PREFIXES: Tuple[str, str] = ('fwd:', 'forwarded:')

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("GmailSearchAPI")

# --- Global Variables / State (Simple Caching) ---
# In a production scenario, consider more robust state management
# or FastAPI lifespan events for initializing the service.
gmail_service_instance: Optional[Resource] = None
service_lock = asyncio.Lock() # Prevent race conditions during auth checks

# --- Pydantic Models ---

class SearchRequest(BaseModel):
    """Request model for the email search endpoint."""
    sender: str = Field(..., description="Email address of the sender to filter by.")
    recipient: EmailStr = Field(..., description="Email address of the recipient (e.g., the authenticated user).")
    time_frame_minutes: int = Field(10, gt=0, description="Search duration going back from now (in minutes).")
    results_limit: int = Field(3, gt=0, description="Maximum number of email results to return.")

class FoundEmail(BaseModel):
    """Model representing a found and saved email."""
    filename: str
    subject: str
    sender: str

class SearchResponse(BaseModel):
    """Response model containing the paths of saved email files."""
    message: str
    search_criteria: SearchRequest
    saved_files: List[str] = Field(default_factory=list)

# --- Helper Classes ---

class EmailData:
    """Represents extracted data for a single email."""
    def __init__(self, msg_id: str, subject: str, sender: str, date_utc: Optional[datetime],
                 delivered_to: str, recipients: str, is_forwarded: bool,
                 body: str, headers: Dict[str, str]):
        self.id: str = msg_id
        self.subject: str = subject
        self.sender: str = sender # Raw sender string from header
        self.sender_email: Optional[str] = email.utils.parseaddr(sender)[0] # Parsed sender email
        self.date_utc: Optional[datetime] = date_utc # Parsed date as UTC datetime
        self.date_str: str = date_utc.strftime('%Y-%m-%d %H:%M:%S %Z') if date_utc else "(No Date)"
        self.delivered_to: str = delivered_to
        self.recipients: str = recipients # Raw recipients string
        self.recipient_emails: List[str] = [addr[1] for addr in email.utils.getaddresses([recipients])] # Parsed recipient emails
        self.is_forwarded: bool = is_forwarded
        self.body: str = body # This will be HTML or plain text wrapped in <pre>
        self.headers: Dict[str, str] = headers

    def __repr__(self) -> str:
        return (f"EmailData(id='{self.id}', subject='{self.subject[:30]}...', "
                f"sender='{self.sender_email}', date='{self.date_str}')")

# --- Core Functions ---

async def get_gmail_service(
    token_path: str = DEFAULT_TOKEN_PATH,
    credentials_path: str = DEFAULT_CREDENTIALS_PATH
) -> Resource:
    """
    Authenticates with Gmail API for server use (loads/refreshes token).
    Raises HTTPException if authentication fails.
    Uses a simple global cache and lock for the service instance.
    """
    global gmail_service_instance
    async with service_lock:
        # Check cache first
        if gmail_service_instance and gmail_service_instance._http.credentials.valid:
             logger.debug("Returning cached Gmail service instance.")
             return gmail_service_instance

        logger.info("Attempting to authenticate/refresh Gmail service.")
        creds: Optional[Credentials] = None
        # Use the current values of the global paths
        current_creds_path = DEFAULT_CREDENTIALS_PATH
        current_token_path = DEFAULT_TOKEN_PATH

        if not os.path.exists(current_creds_path):
            logger.error(f"CRITICAL: Credentials file not found at: {current_creds_path}")
            raise HTTPException(status_code=500, detail="Server configuration error: Missing credentials file.")

        if os.path.exists(current_token_path):
            try:
                with open(current_token_path, 'r') as token_file:
                    creds = Credentials.from_authorized_user_info(json.load(token_file), SCOPES)
                logger.debug(f"Loaded credentials from {current_token_path}")
            except Exception as e:
                logger.error(f"Failed to load token from {current_token_path}: {e}")
                creds = None

        # Refresh if expired
        if creds and creds.expired and creds.refresh_token:
            logger.info("Refreshing expired credentials.")
            try:
                creds.refresh(Request())
                # Save the refreshed credentials
                with open(current_token_path, 'w') as token_file:
                    token_file.write(creds.to_json())
                logger.info(f"Saved refreshed credentials to {current_token_path}")
            except Exception as e:
                logger.error(f"Failed to refresh token: {e}. Manual re-authentication might be required.")
                if os.path.exists(current_token_path):
                    try:
                        os.remove(current_token_path)
                        logger.warning(f"Removed potentially invalid token file: {current_token_path}")
                    except OSError as rm_err:
                        logger.error(f"Error removing token file {current_token_path}: {rm_err}")
                raise HTTPException(status_code=503, detail="Gmail authentication failed (token refresh error).")

        # Check if valid credentials exist now
        if not creds or not creds.valid:
            logger.error(f"No valid credentials available. Ensure '{current_token_path}' exists and is valid (run script once interactively?).")
            raise HTTPException(status_code=503, detail="Gmail authentication failed: No valid token found.")

        # Build and cache the service
        try:
            service = build('gmail', 'v1', credentials=creds)
            gmail_service_instance = service # Cache the instance
            logger.info("Gmail API service built successfully.")
            return service
        except Exception as e:
            logger.error(f"Failed to build Gmail service: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to build Gmail service.")

# --- Email Parsing and Processing Functions (Adapted from original script) ---

def _parse_headers(headers_list: List[Dict[str, str]]) -> Dict[str, str]:
    """Parses the list of header dictionaries into a single dictionary."""
    header_dict: Dict[str, str] = {}
    for header in headers_list:
        name = header.get('name', '').lower()
        value = header.get('value', '')
        if name:
            header_dict[name] = value
    return header_dict

def _find_message_body(part: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """Recursively searches message parts for the best available body content."""
    mime_type = part.get('mimeType')
    body = part.get('body', {})
    data = body.get('data')

    if mime_type in PREFERRED_MIME_TYPES and data:
        try:
            content = base64.urlsafe_b64decode(data).decode('utf-8', errors='replace')
            return content, mime_type
        except Exception as e:
            logger.warning(f"Could not decode body part with MIME type {mime_type}: {e}")
            return None, None

    if mime_type and mime_type.startswith('multipart/') and 'parts' in part:
        for preferred_type in PREFERRED_MIME_TYPES:
            for subpart in part['parts']:
                content, found_mime_type = _find_message_body(subpart)
                if content and found_mime_type == preferred_type:
                    return content, found_mime_type
        for subpart in part['parts']:
             content, found_mime_type = _find_message_body(subpart)
             if content:
                 return content, found_mime_type
    return None, None

def _check_forwarded(subject: str, body: Optional[str], headers: Dict[str, str]) -> bool:
    """Checks if an email appears to be forwarded."""
    if any(subject.lower().startswith(prefix) for prefix in FORWARDED_SUBJECT_PREFIXES):
        return True
    if body and any(indicator in body for indicator in FORWARDED_INDICATORS):
        return True
    return False

def _parse_date_to_utc(date_str: str) -> Optional[datetime]:
    """Parses date string and returns aware datetime object in UTC."""
    try:
        parsed_dt = email.utils.parsedate_to_datetime(date_str)
        if parsed_dt:
            # If naive, assume UTC (common in email headers)
            if parsed_dt.tzinfo is None or parsed_dt.tzinfo.utcoffset(parsed_dt) is None:
                 return parsed_dt.replace(tzinfo=timezone.utc)
            # Convert aware datetime to UTC
            return parsed_dt.astimezone(timezone.utc)
        return None
    except Exception as e:
        logger.warning(f"Could not parse date string '{date_str}': {e}. Treating as None.")
        return None

def parse_email_message(msg: Dict[str, Any]) -> Optional[EmailData]:
    """Parses a raw Gmail message object into an EmailData object."""
    try:
        payload = msg.get('payload', {})
        headers_list = payload.get('headers', [])
        headers = _parse_headers(headers_list)

        subject = headers.get('subject', '(No Subject)')
        sender = headers.get('from', '(No Sender)')
        date_raw = headers.get('date', '(No Date)')
        delivered_to = headers.get('delivered-to', '(Not Available)')
        recipients = headers.get('to', '(Not Available)') # Includes To, Cc, Bcc potentially

        body_content, mime_type = _find_message_body(payload)
        is_forwarded = _check_forwarded(subject, body_content, headers)

        display_body = "<p>(No content available)</p>"
        if body_content:
            if mime_type == 'text/plain':
                escaped_content = body_content.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                display_body = f"<pre>{escaped_content}</pre>"
            elif mime_type == 'text/html':
                display_body = body_content
            else:
                display_body = "<p>(Unsupported content type)</p>"

        parsed_date_utc = _parse_date_to_utc(date_raw)

        return EmailData(
            msg_id=msg['id'],
            subject=subject,
            sender=sender,
            date_utc=parsed_date_utc,
            delivered_to=delivered_to,
            recipients=recipients,
            is_forwarded=is_forwarded,
            body=display_body,
            headers=headers
        )
    except Exception as e:
        logger.error(f"Failed to parse email message ID {msg.get('id', 'N/A')}: {e}", exc_info=True)
        return None

# --- File Saving Functions ---

def create_safe_filename(base_dir: str, email_id: str, subject: str) -> str:
    """Creates a safe relative filename for the email."""
    safe_subject = re.sub(r'[\\/*?:"<>|]', "", subject)
    safe_subject = re.sub(r'\s+', '_', safe_subject)
    safe_subject = re.sub(r'_+', '_', safe_subject)
    safe_subject = safe_subject[:60].strip('_')
    if not safe_subject:
        safe_subject = "no_subject"
    # Return relative path
    return f"email_{email_id}_{safe_subject}.html"

def generate_html_content(email_data: EmailData) -> str:
    """Generates the HTML content for saving the email."""
    # (Using the same HTML template as before)
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{email_data.subject}</title>
    <style>
        body {{ font-family: 'Segoe UI', Roboto, Arial, sans-serif; margin: 0; padding: 0; background-color: #f6f8fc; color: #202124; }}
        .email-container {{ max-width: 800px; margin: 20px auto; background-color: #ffffff; border: 1px solid #dadce0; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
        .email-header {{ padding: 20px 24px; border-bottom: 1px solid #e0e0e0; background-color: #f1f3f4; }}
        .email-subject {{ font-size: 1.3em; font-weight: 500; color: #1f1f1f; margin-bottom: 16px; line-height: 1.4; }}
        .email-meta {{ display: grid; grid-template-columns: 100px auto; gap: 8px 12px; color: #5f6368; font-size: 0.9em; }}
        .email-label {{ font-weight: 600; text-align: right; color: #3c4043; }}
        .email-value {{ word-break: break-word; }}
        .email-forwarded {{ background-color: #fff0c7; color: #7a5b00; padding: 3px 8px; border-radius: 4px; display: inline-block; font-size: 0.85em; font-weight: 500; margin-top: 5px; grid-column: 2 / 3; }}
        .email-content {{ padding: 24px; line-height: 1.6; font-size: 1em; word-wrap: break-word; }}
        .email-content p {{ margin: 0 0 1em 0; }}
        .email-content a {{ color: #1a73e8; text-decoration: none; }}
        .email-content a:hover {{ text-decoration: underline; }}
        .email-content img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .email-content pre {{ white-space: pre-wrap; word-wrap: break-word; background-color: #f1f3f4; padding: 12px; border-radius: 4px; font-family: 'Courier New', Courier, monospace; font-size: 0.95em; overflow-x: auto; }}
    </style>
</head>
<body>
    <div class="email-container">
        <div class="email-header">
            <div class="email-subject">{email_data.subject}</div>
            <div class="email-meta">
                <div class="email-label">From:</div>
                <div class="email-value">{email_data.sender}</div>
                <div class="email-label">Date:</div>
                <div class="email-value">{email_data.date_str}</div>
                <div class="email-label">To:</div>
                <div class="email-value">{email_data.recipients}</div>
                <div class="email-label">Delivered To:</div>
                <div class="email-value">{email_data.delivered_to}</div>
                {f'<div class="email-label">Status:</div><div class="email-value"><span class="email-forwarded">Forwarded</span></div>' if email_data.is_forwarded else ''}
            </div>
        </div>
        <div class="email-content">
            {email_data.body}
        </div>
    </div>
</body>
</html>"""
    return html_template

async def search_and_save_emails(
    service: Resource,
    criteria: SearchRequest,
    output_dir: str # Use the value passed from the endpoint
) -> List[FoundEmail]: # *** Ensure return type hint is List[FoundEmail] ***
    """
    Searches emails based on criteria, saves matches, and returns a list
    of FoundEmail objects.
    """
    # *** Initialize list to hold FoundEmail objects ***
    found_emails_list: List[FoundEmail] = []
    found_count = 0


    # 1. Calculate time window
    now = datetime.now(timezone.utc)
    start_time_utc = now - timedelta(minutes=criteria.time_frame_minutes)
    # Gmail API uses Unix timestamp (seconds) for 'after'
    start_timestamp = int(start_time_utc.timestamp())

    # 2. Construct Gmail search query
    query_parts = [
        f'from:"{criteria.sender}"',
        f'to:"{criteria.recipient}"',
        f'after:{start_timestamp}'
    ]
    query = " ".join(query_parts)
    logger.info(f"Executing Gmail search with query: {query}")

    try:

        # 3. List message IDs matching the query
        list_request = service.users().messages().list(
            userId='me',
            q=query,
            maxResults=criteria.results_limit * 2 # Fetch a bit more for filtering
        )
        response = list_request.execute()
        messages = response.get('messages', [])

        if not messages:
            logger.info('No messages found matching the initial query.')
            return []

        logger.info(f"Found {len(messages)} potential messages. Fetching details and filtering...")

        # Ensure output directory exists

        # 4. Fetch details, parse, filter, and save
        for message_ref in messages:
            if found_count >= criteria.results_limit:
                break

            msg_id = message_ref['id']
            try:
                msg = service.users().messages().get(
                    userId='me', id=msg_id, format='full'
                ).execute()

                email_data = parse_email_message(msg)

                if not email_data:
                    logger.warning(f"Skipping message {msg_id} due to parsing error.")
                    continue

                # 5. Apply Strict Filtering
                print(email_data.sender_email)
                sender_match = email_data.sender_email and email_data.sender_email.lower() == criteria.sender.lower()
                recipient_match = any(r.lower() == criteria.recipient.lower() for r in email_data.recipient_emails)
                date_match = email_data.date_utc and email_data.date_utc >= start_time_utc

                if sender_match and recipient_match and date_match:
                    logger.debug(f"Message {msg_id} matches all criteria. Saving...")
                    html_content = generate_html_content(email_data)
                    relative_filename = create_safe_filename(output_dir, email_data.id, email_data.subject)
                    full_filepath = os.path.join(output_dir, relative_filename)

                    try:
                        with open(full_filepath, 'w', encoding='utf-8') as f:
                            f.write(html_content)
                        logger.info(f"Saved email to {full_filepath}")

                        # *** THIS IS THE CRITICAL FIX ***
                        # Create a FoundEmail object and append it
                        found_email_obj = FoundEmail(
                            filename=relative_filename,
                            subject=email_data.subject if email_data.subject else "(No Subject)", # Handle potential None subject
                            sender=email_data.sender if email_data.sender else "(No Sender)" # Handle potential None sender
                        )
                        found_emails_list.append(found_email_obj)
                        # *** DO NOT append relative_filename directly ***
                        # Example of WRONG way: found_emails_list.append(relative_filename)

                        found_count += 1
                    except IOError as e:
                        logger.error(f"Could not write file {full_filepath}: {e}")
                # else: # Optional: keep the debug log if needed
                #      logger.debug(f"Skipping message {msg_id}. Failed strict filter...")

            except HttpError as http_err:
                 logger.error(f"HTTP error fetching/processing message {msg_id}: {http_err}")
            except Exception as e:
                logger.error(f"Error processing message {msg_id}: {e}", exc_info=True)

        logger.info(f"Search complete. Found and saved {found_count} emails matching criteria.")
        # *** Ensure you return the list of objects ***
        return found_emails_list

    except HttpError as error:
        logger.error(f'An API error occurred: {error}')
        raise HTTPException(status_code=503, detail=f"Gmail API error: {error}")
    except Exception as e:
        logger.error(f"An unexpected error occurred during search: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error during email search.")


# --- 

app = FastAPI(
    title="Gmail Email Search API",
    description="Searches Gmail for recent emails based on criteria and saves them.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],         # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],         # Allow all HTTP methods
    allow_headers=["*"],         # Allow all headers
)

app.mount("/static", StaticFiles(directory=DEFAULT_OUTPUT_DIR), name="static_emails")



class SearchResponse(BaseModel):
    message: str
    search_criteria: SearchRequest
    # Change saved_files to found_emails
    found_emails: List[FoundEmail] = Field(default_factory=list)


# --- API Endpoints ---

@app.post("/search-emails", response_model=SearchResponse)
async def search_emails_endpoint(
    search_request: SearchRequest,
    service: Resource = Depends(get_gmail_service) # Inject authenticated service
) ->  List[FoundEmail]:
    """
    Searches for emails matching sender, recipient, and time frame,
    saves them as HTML, and returns the list of relative file paths.
    """
    logger.info(f"Received search request: {search_request.dict()}")

    # Use the current value of the global output directory
    output_directory = DEFAULT_OUTPUT_DIR

    try:
        # Change variable name
        found_emails_result = await search_and_save_emails(
            service=service,
            criteria=search_request,
            output_dir=output_directory
        )

        if found_emails_result:
             message = f"Successfully found and saved {len(found_emails_result)} emails."
        else:
             message = "No emails found matching the specified criteria."

        return SearchResponse(
            message=message,
            search_criteria=search_request,
            # Assign to the correct response field
            found_emails=found_emails_result
        )
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Unhandled error in search endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal server error occurred.")


@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "ok"}

# --- Main Execution (for running the server) ---

def main():
    """Main function to parse args and run the FastAPI server."""
    parser = argparse.ArgumentParser(
        description="Run the FastAPI Gmail Search Server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # Add arguments for server config
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("-p", "--port", type=int, default=8000, help="Port number")
    parser.add_argument("-o", "--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Directory for saved emails")
    parser.add_argument("-t", "--token", type=str, default=DEFAULT_TOKEN_PATH, help="Path to token.json")
    parser.add_argument("-s", "--secrets", type=str, default=DEFAULT_CREDENTIALS_PATH, help="Path to credentials.json")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    # Set logging level FIRST
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger().setLevel(logging.DEBUG) # Set root logger level too
        logger.info("Verbose logging enabled.")
    else:
         logger.setLevel(logging.INFO)
         logging.getLogger().setLevel(logging.INFO)

    # *** CORRECTED PLACEMENT OF GLOBAL DECLARATION ***
    # Declare globals *after* parsing args and setting logging, but *before* using them
    # Now update the global variables from args
    DEFAULT_OUTPUT_DIR = args.output_dir
    DEFAULT_TOKEN_PATH = args.token
    DEFAULT_CREDENTIALS_PATH = args.secrets

    # Now it's safe to use the globals in logging etc.
    logger.info(f"Starting server on {args.host}:{args.port}")
    logger.info(f"Using Token: {DEFAULT_TOKEN_PATH}, Secrets: {DEFAULT_CREDENTIALS_PATH}")
    logger.info(f"Output directory: {DEFAULT_OUTPUT_DIR}")
    logger.warning("Ensure 'token.json' exists and is valid before starting.")

    # Run the Uvicorn server
    uvicorn.run(
        "__main__:app", # Reference the app instance within the current module
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level= "debug" if args.verbose else "info"
    )

if __name__ == '__main__':
    main()
