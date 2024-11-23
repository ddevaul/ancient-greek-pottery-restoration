import requests
import csv
from xml.etree import ElementTree as ET

def scrape_metadata(api_url, output_csv_path):
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()

        # Parse XML response
        root = ET.fromstring(response.text)

        # Prepare CSV file
        with open(output_csv_path, mode="w", newline="") as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow(["URI", "Technique", "Shape"])

            # Parse records and write to CSV
            for record in root.findall("Record"):
                uri = record.find("URI").text.strip()
                technique = record.find("Technique").text.strip()
                shape = record.find("Shape").text.strip()

                writer.writerow([uri, technique, shape])
        print(f"CSV saved to {output_csv_path}")

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch catalog: {e}")
    except ET.ParseError as e:
        print(f"Error parsing XML: {e}")


