import os
import base64
import json
import re
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
import email
from email.mime.text import MIMEText
from datetime import datetime
import random

SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

def get_gmail_service():
    """Get an authorized Gmail API service instance."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_info(
            json.loads(open('token.json', 'r').read())
        )
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    service = build('gmail', 'v1', credentials=creds)
    return service

def get_html_emails(count=2):
    """Get the most recent emails from Gmail inbox with HTML content and all headers.
    
    Args:
        count: Number of emails to retrieve (default: 2)
        
    Returns:
        List of email details including HTML content and all headers
    """
    service = get_gmail_service()
    
    # Get messages from inbox
    results = service.users().messages().list(
        userId='me',
        labelIds=['INBOX'],
        maxResults=count
    ).execute()
    
    messages = results.get('messages', [])
    
    if not messages:
        print('No messages found.')
        return []
        
    html_emails = []
    
    for message in messages:
        msg = service.users().messages().get(
            userId='me', 
            id=message['id'],
            format='full'
        ).execute()
        
        # Extract all headers
        headers = msg['payload']['headers']
        header_dict = {}
        
        # Process all headers into a dictionary
        for header in headers:
            name = header['name'].lower()
            value = header['value']
            header_dict[name] = value
        
        # Extract specific headers
        subject = header_dict.get('subject', '(No subject)')
        sender = header_dict.get('from', '(No sender)')
        date = header_dict.get('date', '(No date)')
        delivered_to = header_dict.get('delivered-to', '(Not available)')
        to = header_dict.get('to', '(Not available)')
        
        # Check for forwarded indicators in headers or content
        forwarded = False
        if 'forwarded' in subject.lower() or 'fwd:' in subject.lower():
            forwarded = True
        
        # Get email HTML content
        html_content = ""
        
        # Function to extract HTML parts recursively
        def extract_html_parts(part):
            nonlocal forwarded

            if part.get('mimeType') == 'text/html' and 'data' in part.get('body', {}):
                content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                # Check for forwarded content in the HTML
                if not forwarded and ('---------- Forwarded message ---------' in content or 
                                     '-------- Original Message --------' in content):
                    forwarded = True
                return content
            
            if 'parts' in part:
                for subpart in part['parts']:
                    html = extract_html_parts(subpart)
                    if html:
                        return html
            return ""
        
        # Try to find HTML content
        if 'parts' in msg['payload']:
            for part in msg['payload']['parts']:
                html_content = extract_html_parts(part)
                if html_content:
                    break
        elif msg['payload'].get('mimeType') == 'text/html' and 'data' in msg['payload'].get('body', {}):
            html_content = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
            # Check for forwarded content
            if not forwarded and ('---------- Forwarded message ---------' in html_content or 
                                 '-------- Original Message --------' in html_content):
                forwarded = True
        
        # If no HTML content found, try to get plain text
        if not html_content:
            plain_text = ""
            
            def extract_plain_text(part):
                nonlocal forwarded

                if part.get('mimeType') == 'text/plain' and 'data' in part.get('body', {}):
                    content = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                    # Check for forwarded content in plain text
                    if not forwarded and ('---------- Forwarded message ---------' in content or 
                                         '-------- Original Message --------' in content):
                        forwarded = True
                    return content
                
                if 'parts' in part:
                    for subpart in part['parts']:
                        text = extract_plain_text(subpart)
                        if text:
                            return text
                return ""
            
            if 'parts' in msg['payload']:
                for part in msg['payload']['parts']:
                    plain_text = extract_plain_text(part)
                    if plain_text:
                        break
            elif msg['payload'].get('mimeType') == 'text/plain' and 'data' in msg['payload'].get('body', {}):
                plain_text = base64.urlsafe_b64decode(msg['payload']['body']['data']).decode('utf-8')
                # Check for forwarded content
                if not forwarded and ('---------- Forwarded message ---------' in plain_text or 
                                     '-------- Original Message --------' in plain_text):
                    forwarded = True
            
            if plain_text:
                # Convert plain text to simple HTML
                html_content = f"<pre>{plain_text}</pre>"
            else:
                html_content = "<p>(No content available)</p>"
        
        # Format date
        try:
            parsed_date = email.utils.parsedate_to_datetime(date)
            formatted_date = parsed_date.strftime('%Y-%m-%d %H:%M:%S')
        except:
            formatted_date = date
        
        email_data = {
            'id': message['id'],
            'subject': subject,
            'sender': sender,
            'date': formatted_date,
            'delivered_to': delivered_to,
            'to': to,
            'forwarded': forwarded,
            'html_content': html_content
        }
        
        html_emails.append(email_data)
    
    return html_emails

def create_safe_filename(subject):
    """Create a safe filename from the email subject."""
    # Remove invalid filename characters
    safe_subject = re.sub(r'[\\/*?:"<>|]', "", subject)
    # Replace spaces with underscores
    safe_subject = safe_subject.replace(' ', '_')
    # Limit length to avoid filename too long errors
    safe_subject = safe_subject[:50]
    # If subject is empty or just spaces, use "no_subject"
    if not safe_subject or safe_subject.isspace():
        safe_subject = "no_subject"
    return f"./extracted_mails/email_{random.randint(1,99)}_{safe_subject}"

def save_emails_as_html(emails):
    """Save emails as HTML files with Gmail-like styling."""
    for email_data in emails:
        # Create safe filename based on subject
        filename = create_safe_filename(email_data['subject']) + ".html"
        
        # Create Gmail-like styling
        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>{email_data['subject']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f6f6f6; }}
        .email-container {{ max-width: 800px; margin: 20px auto; background-color: white; border: 1px solid #e0e0e0; border-radius: 8px; overflow: hidden; }}
        .email-header {{ padding: 16px; border-bottom: 1px solid #e0e0e0; }}
        .email-subject {{ font-size: 20px; font-weight: bold; color: #202124; margin-bottom: 12px; }}
        .email-meta {{ display: flex; flex-direction: column; gap: 6px; color: #5f6368; font-size: 14px; }}
        .email-row {{ display: flex; }}
        .email-label {{ width: 100px; font-weight: bold; }}
        .email-value {{ flex: 1; }}
        .email-forwarded {{ background-color: #fef9e7; padding: 4px 8px; border-radius: 4px; display: inline-block; }}
        .email-content {{ padding: 16px; }}
        .email-content img {{ max-width: 100%; }}
    </style>
</head>
<body>
    <div class="email-container">
        <div class="email-header">
            <div class="email-subject">{email_data['subject']}</div>
            <div class="email-meta">
                <div class="email-row">
                    <div class="email-label">From:</div>
                    <div class="email-value">{email_data['sender']}</div>
                </div>
                <div class="email-row">
                    <div class="email-label">Date:</div>
                    <div class="email-value">{email_data['date']}</div>
                </div>
                <div class="email-row">
                    <div class="email-label">To:</div>
                    <div class="email-value">{email_data['to']}</div>
                </div>
                <div class="email-row">
                    <div class="email-label">Delivered To:</div>
                    <div class="email-value">{email_data['delivered_to']}</div>
                </div>
                {f'<div class="email-row"><div class="email-label"></div><div class="email-value"><span class="email-forwarded">Forwarded</span></div></div>' if email_data['forwarded'] else ''}
            </div>
        </div>
        <div class="email-content">
            {email_data['html_content']}
        </div>
    </div>
</body>
</html>"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"Email saved as {filename}")

def main():
    """Main function to get and display recent emails."""
    emails = get_html_emails(5)
    
    if emails:
        print(f"Retrieved {len(emails)} recent emails")
        save_emails_as_html(emails)
    else:
        print("No emails were retrieved")

if __name__ == '__main__':
    main()