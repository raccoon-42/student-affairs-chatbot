import pdfplumber
import re
from datetime import datetime
import json

def parse_date(date_str):
    """Parse Turkish date format into datetime object"""
    try:
        # Handle Turkish month abbreviations
        turkish_months = {
            'Oca': 'January', 'Şub': 'February', 'Mar': 'March',
            'Nis': 'April', 'May': 'May', 'Haz': 'June',
            'Tem': 'July', 'Ağu': 'August', 'Eyl': 'September',
            'Eki': 'October', 'Kas': 'November', 'Ara': 'December'
        }
        
        # Remove day name if present
        date_str = re.sub(r'\s+[PÇCPSPÇC]\w+$', '', date_str)
        
        # Split the date string
        parts = date_str.split('.')
        if len(parts) != 3:
            return None
            
        day, month, year = parts
        
        # Convert Turkish month abbreviation to English
        month = turkish_months.get(month, month)
        
        # Convert 2-digit year to 4-digit year
        year = '20' + year if int(year) < 50 else '19' + year
        
        # Create date string in standard format
        date_str = f"{day} {month} {year}"
        
        return datetime.strptime(date_str, "%d %B %Y")
    except:
        return None

def extract_event_type(text):
    """Extract event type based on text content"""
    text = text.lower()
    
    # Check for specific event types
    if "son gün" in text:
        return "deadline"
    elif "arasında" in text:
        return "period"
    elif "tatili" in text or "bayram" in text:
        return "holiday"
    elif "sınav" in text:
        return "exam"
    elif "kayıt" in text:
        return "registration"
    elif "ders" in text:
        return "course"
    elif "mezuniyet" in text:
        return "graduation"
    elif "başvuru" in text:
        return "application"
    elif "duyuru" in text or "ilan" in text:
        return "announcement"
    
    return "event"

def extract_academic_period(text):
    """Extract academic period from text"""
    text = text.lower()
    
    # Check for specific academic periods
    if "güz yarıyılı" in text:
        return "fall"
    elif "bahar yarıyılı" in text:
        return "spring"
    elif "yaz öğretimi" in text:
        return "summer"
    
    return None

def extract_schedule_table(pdf_path, output_file):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_file, 'w', encoding='utf-8') as file:
            current_section = None
            current_subsection = None
            
            for page in pdf.pages:
                # Extract table is designed for Academic Calendar
                tables = page.extract_tables(table_settings = {
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "snap_tolerance": 3,
                    "snap_x_tolerance": 3,
                    "snap_y_tolerance": 3,
                    "join_tolerance": 3,
                    "join_x_tolerance": 3,
                    "join_y_tolerance": 3,
                    "edge_min_length": 3,
                    "min_words_vertical": 3,
                    "min_words_horizontal": 1,
                    "intersection_tolerance": 3,
                    "intersection_x_tolerance": 3,
                    "intersection_y_tolerance": 3,
                    "text_tolerance": 3,
                    "text_x_tolerance": 1,
                    "text_y_tolerance": 3,
                })

                for table in tables:
                    for row in table:
                        # Skip empty rows
                        if not any(row):
                            continue
                            
                        # Check if this is a section header (spans all columns)
                        if len(row) == 3 and not row[0] and not row[1] and row[2]:
                            # Update section hierarchy
                            if current_section is None:
                                current_section = row[2].strip()
                                current_subsection = None
                            else:
                                current_subsection = row[2].strip()
                            continue
                            
                        # Process regular event rows
                        if len(row) == 3:
                            first_column, second_column, event = row
                            
                            # Skip if no event text
                            if not event:
                                continue
                                
                            # Prepare metadata
                            metadata = {
                                "section": current_section,
                                "subsection": current_subsection,
                                "event_type": extract_event_type(event),
                                "academic_period": extract_academic_period(event)
                            }
                            
                            # Handle date range
                            if first_column and second_column:
                                start_date = parse_date(first_column)
                                end_date = parse_date(second_column)
                                if start_date and end_date:
                                    metadata["start_date"] = start_date.isoformat()
                                    metadata["end_date"] = end_date.isoformat()
                                    natural_language_output = f"{first_column} ve {second_column} tarihleri arasında {event.lower()}"
                                else:
                                    continue
                            # Handle single date
                            elif first_column:
                                date = parse_date(first_column)
                                if date:
                                    metadata["start_date"] = date.isoformat()
                                    metadata["end_date"] = date.isoformat()
                                    natural_language_output = f"{first_column} tarihinde {event.lower()}"
                                else:
                                    continue
                            else:
                                continue
                            
                            # Write to file with metadata
                            file.write(f"{natural_language_output}\t{json.dumps(metadata, ensure_ascii=False)}\n")
                        else:
                            print(f"Unexpected row format: {row}")

if __name__ == "__main__":
    pdf_path = "schedule.pdf"
    output_file = "schedule_text.txt"
    extract_schedule_table(pdf_path, output_file)