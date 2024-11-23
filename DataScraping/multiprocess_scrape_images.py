import csv
import requests
from multiprocessing import Pool, cpu_count
import xml.etree.ElementTree as ET
from pathlib import Path
import csv
from typing import Union
import math


def parse_xml(xml_url: str) -> list[str]:
    try:
        response = requests.get(xml_url, timeout=10)
        response.raise_for_status()  # Raise an exception for errors

        text = response.text.replace("&", "")
        root = ET.fromstring(text)
        records = root.findall("Record")

        im_file_names = []
        for record in records:
            image_records = record.findall("Image-Record")
            for ir in image_records:
                file_names = ir.findall("Filename")
                for f in file_names:
                    im_file_names.append(f.text.strip())
        return im_file_names

    except requests.exceptions.RequestException as e:
        print(f"Failed to fetch XML from {xml_url}: {e}")
    except ET.ParseError as e:
        print(f"Failed to parse XML from {xml_url}: {e}")
    except Exception as e:
        print(f"Unexpected error while processing XML from {xml_url}: {e}")

    return []


def get_images(
    im_file_names: list[str],
    technique: str,
    shape: str,
    og_url: str,
    image_folder: Path,
    output_folder: Path,
    base_url: str = "https://www.carc.ox.ac.uk/Vases/SPIFF",
) -> None:

    image_folder.mkdir(parents=True, exist_ok=True)
    for im in im_file_names:
        im = im.strip()
        
        im_name = Path(im).stem
        file_name = image_folder / f"{im_name}-{technique}.jpg"
        if Path(file_name).exists():
            # print("Skipping image as it already exists")
            continue
        new_url = f"{base_url}/{im}cc001001.jpe"
        try:
            
            # Download the Image
            response = requests.get(new_url, timeout=10)
            response.raise_for_status()  # Raise an exception for errors
            im_byte_code = response.content

            # im_name = Path(im).stem
            # file_name = image_folder / f"{im_name}-{technique}.jpg"

            with file_name.open("wb") as f:
                f.write(im_byte_code)

            with output_folder.open("a") as new_file:
                new_file.write(f"{file_name}, {og_url}, {technique}, {shape}\n")
            # print(f"Downloaded and saved {file_name}")

        except requests.exceptions.RequestException as e:
            print(f"Failed to download image from {new_url}: {e}")
        except Exception as e:
            print(f"Unexpected error while processing image {im}: {e}")


def process_batch(
    batch: list[list[str]], image_folder: Path, output_folder: Path, error_file: Path, batch_id: int
) -> None:
    """
    Process a batch of rows: parse XML and download images.
    """
    print(f"Starting Batch: {batch_id}")

    for row in batch:
        try:
            if row[0] == "\ufeffURI":
                continue  # Skip header

            technique = row[3].strip()
            shape = row[5].strip()

            if technique == "BLACK-FIGURE" and shape in {"AMPHORA, NECK", "LEKYTHOS"}:
                xml_url = f"{row[0].strip()}/xml"
                im_file_names = parse_xml(xml_url)  # Get the File names.
                shape = "AMPHORA NECK" if shape == "AMPHORA, NECK" else shape
                get_images(
                    im_file_names,
                    technique,
                    shape,
                    row[0].strip(),
                    image_folder,
                    output_folder,
                )
        except Exception as e:
            print(f"Error processing row {row[0]}: {e}")
            with error_file.open("a") as ef:
                ef.write(f"{row[0]}\n")
    print(f"batch_id: {batch_id} Completed")


def multiprocess_scrape_images(
    data_file: Union[str, Path],
    image_folder: Union[str, Path],
    output_folder: Union[str, Path],
    error_file: Union[str, Path],
    num_processes: int = 4,
    batch_size: int = 1000,
) -> None:
    
    
    data_file = Path(data_file)
    image_folder = Path(image_folder)
    output_folder = Path(output_folder)
    error_file = Path(error_file)

    print(f"Getting data from: {data_file}")
    print(f"Saving images to: {image_folder}")
    print(f"Saving new metadata to: {output_folder}")

    image_folder.mkdir(parents=True, exist_ok=True)

    # Read all the rows of data (hopefully this can fit in memory)
    with data_file.open("r") as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        rows = list(reader)

    # Batch the rows
    num_batches = math.ceil(len(rows) / batch_size)
    batches = [rows[i * batch_size : (i + 1) * batch_size] for i in range(num_batches)]

    # Pass each process a batch
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            process_batch,
            [(batch, image_folder, output_folder, error_file, batch_id) for batch_id, batch in enumerate(batches)],
        )
