import pdfplumber

def extract_schedule_table(pdf_path, output_file):
    with pdfplumber.open(pdf_path) as pdf:
        with open(output_file, 'w', encoding='utf-8') as file:
            for page in pdf.pages:
                # Extract table is designed for Academic Calendar
                tables = page.extract_tables(table_settings = {
                    "vertical_strategy": "lines",  # Default: "lines", can try "text" if lines are missing
                    "horizontal_strategy": "lines",  # Default: "lines", can try "text" for better alignment
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
                        if len(row) == 3:
                            first_column, second_column, event = row
                            # Reformat into human-like text for the chatbot
                            if first_column and second_column and event:
                                natural_language_output = f"{first_column} ve {second_column} tarihleri arasında {event.lower()}"
                                file.write(natural_language_output + "\n")

                            elif first_column and event:
                                natural_language_output = f"{first_column} tarihinde {event.lower()}"
                                file.write(natural_language_output + "\n")


                        else:
                            print(f"Tarihsiz Satır veya Genel Başlık: {row}")

pdf_path = "schedule.pdf"
output_file = "schedule_text.txt"
extract_schedule_table(pdf_path, output_file)