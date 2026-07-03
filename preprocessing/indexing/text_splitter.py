import re
import argparse

from preprocessing.extraction import parse_date, extract_event_type, extract_academic_period, is_date_line


def split_text(file_path):
    """Split an academic calendar text file into per-event chunks with metadata."""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    lines = text.split('\n')
    chunks = []
    current_event = []
    current_date_info = None
    current_period = None

    def add_chunk():
        """Helper function to add an event chunk while preventing duplicates."""
        nonlocal current_event, current_date_info

        if current_event and current_date_info:
            event_text = ' '.join(current_event).strip()

            chunk = {
                "text": event_text,
                "metadata": {
                    "date1": current_date_info['date1'],
                    "date2": current_date_info['date2'],
                    "event": event_text,
                    "event_type": extract_event_type(event_text, default="event"),
                    "academic_period": current_period,
                    "parsed_date1": current_date_info['parsed_date1'],
                    "parsed_date2": current_date_info['parsed_date2']
                }
            }
            chunks.append(chunk)

    for line in lines:
        if not line.strip():
            continue

        # Handle TITLE section
        if line.strip().startswith("TITLE:"):
            add_chunk()  # Save previous event before changing period

            title_text = line.strip().replace("TITLE:", "").strip()
            current_period = extract_academic_period(title_text)

            current_event = []
            current_date_info = None
            continue

        # Check if line contains a date
        if is_date_line(line):
            add_chunk()  # Save previous event before processing a new one

            current_event = []  # Reset event

            date_match = re.match(r'(\d{2}\.[A-Za-zğüşıöçĞÜŞİÖÇ]+\.\d{2})\s+[PÇCPSPÇC]\w+\s+(?:tarihinde|tarihleri arasında)?\s*(.*)', line)
            date_range_match = re.match(r'(\d{2}\.[A-Za-zğüşıöçĞÜŞİÖÇ]+\.\d{2})\s+[PÇCPSPÇC]\w+\s+ve\s+(\d{2}\.[A-Za-zğüşıöçĞÜŞİÖÇ]+\.\d{2})\s+[PÇCPSPÇC]\w+\s+tarihleri arasında\s*(.*)', line)

            if date_range_match:
                date1, date2, event = date_range_match.groups()
                current_date_info = {
                    "text": line.strip(),
                    "date1": date1,
                    "date2": date2,
                    "parsed_date1": parse_date(date1),
                    "parsed_date2": parse_date(date2)
                }
                current_event.append(f"{date1} tarihinden {date2} tarihine kadar")
            elif date_match:
                date1, event = date_match.groups()
                current_date_info = {
                    "text": line.strip(),
                    "date1": date1,
                    "date2": None,
                    "parsed_date1": parse_date(date1),
                    "parsed_date2": None
                }
                current_event.append(f"{date1} tarihinde")

            if event.strip():
                current_event.append(event.strip())
        else:
            current_event.append(line.strip())

    # Process the last event
    add_chunk()

    return chunks


def split_regulation(file_path):
    """Split regulation text into per-article chunks with metadata"""
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into lines first
    lines = text.split('\n')
    chunks = []
    current_article = []
    current_article_info = None
    current_section = None
    current_subsection = None

    def add_chunk():
        if current_article and current_article_info:
            article_text = ' '.join(current_article)
            chunks.append({
                "text": current_article_info['text'] + ' ' + article_text,
                "metadata": {
                    "article_number": current_article_info['article_number'],
                    "article_title": current_article_info['article_title'],
                    "section": current_section,
                    "subsection": current_subsection,
                    "content_type": "regulation",
                    "section_number": current_section.split()[0] if current_section else None,  # e.g., "BİRİNCİ"
                    "section_title": ' '.join(current_section.split()[1:]) if current_section else None,  # e.g., "BÖLÜM"
                    "hierarchy": {
                        "section": current_section,
                        "subsection": current_subsection,
                        "article": current_article_info['article_number']
                    }
                }
            })

    for line in lines:
        # Skip empty lines
        if not line.strip():
            continue

        # Check for section headers
        if line.strip().startswith("BÖLÜM"):
            current_section = line.strip()
            current_subsection = None
            continue

        # Check if line starts with "MADDE" (Article)
        if line.strip().startswith("MADDE"):
            add_chunk()  # Process previous article if exists

            # Start new article
            current_article = []

            # Extract article information
            article_match = re.match(r'MADDE\s+(\d+)\s*–\s*(.*)', line.strip())

            if article_match:
                article_number, article_title = article_match.groups()

                current_article_info = {
                    "text": line.strip(),
                    "article_number": article_number,
                    "article_title": article_title.strip()
                }

                # Add initial article text if exists
                if article_title.strip():
                    current_article.append(article_title.strip())
        else:
            # This is a continuation of the current article
            current_article.append(line.strip())

    # Process the last article if exists
    add_chunk()

    return chunks


def main():
    parser = argparse.ArgumentParser(description='Split text files into chunks with metadata')
    parser.add_argument('input_file', help='Input text file to process')
    parser.add_argument('--show-metadata', action='store_true', help='Show metadata for each chunk')
    parser.add_argument('--type', choices=['calendar', 'regulation'], default='calendar',
                      help='Type of document to process (default: calendar)')

    args = parser.parse_args()

    if args.type == 'calendar':
        chunks = split_text(args.input_file)
    else:
        chunks = split_regulation(args.input_file)

    print(f"Text is split into {len(chunks)} chunks.")

    if args.show_metadata:
        print("\nSample chunks with metadata:")
        for i, chunk in enumerate(chunks):
            print(f"\nChunk {i+1}:")
            print(f"Text: {chunk['text']}")
            print(f"Metadata: {chunk['metadata']}")

if __name__ == "__main__":
    main()
