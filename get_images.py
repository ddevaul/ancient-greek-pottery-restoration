import csv
import requests 
import xml.etree.ElementTree as ET
import os

def parse_xml(xml_url):
  r = requests.get(xml_url)
  text = r.text.replace("&", "")
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
  
def get_images(im_file_names, technique, shape, og_url, image_folder, output_file):
  for im in im_file_names:
    im = im.strip()
    new_url = f"https://www.carc.ox.ac.uk/Vases/SPIFF/{im}cc001001.jpe"
    r = requests.get(new_url)
    im_byte_code = r.content
    im_name = im.split("/")[-2]
    file_name = f"{image_folder}/{im_name}-{technique}.jpg"
    with open(file_name, "wb") as f:
      f.write(im_byte_code)
    with open(output_file, "a+") as new_file:
      new_file.write(f"{file_name}, {og_url}, {technique}, {shape}\n")

def run(data_file, image_folder, output_file, error_file):
  print("Getting data from: ", data_file)
  print("Printing images to: ", image_folder)
  print("Saving new metadata to: ", output_file)
  if not os.path.exists(image_folder):
    print("creating output directory for the images")
    os.mkdir(image_folder)
  with open(data_file, "r") as csvfile:
    reader = csv.reader(csvfile, delimiter=",")
    row_count = 0
    for row in reader:
      if row_count % 100 == 0: # logging
        print(row_count)
      if row[0] != "\ufeffURI":
        technique = row[3]
        shape = row[5]
        if technique.strip() == "BLACK-FIGURE" and (shape.strip() == "AMPHORA, NECK" or shape.strip() == "LEKYTHOS"):
          xml_url = f"{row[0].strip()}/xml"
          im_file_names = parse_xml(xml_url)
          shape = "AMPHORA NECK" if shape == "AMPHORA, NECK" else shape
          try:
            get_images(im_file_names, technique, shape, row[0].strip(), image_folder, output_file)
          except:
            with open(error_file, "+a") as ef:
              ef.write(f"{xml_url}\n")
      row_count += 1

run("data.csv", "./images-nicky-test", "aggregrate_data.csv", "errors.txt")