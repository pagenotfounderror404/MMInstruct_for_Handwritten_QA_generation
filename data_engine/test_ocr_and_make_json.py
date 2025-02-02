import os
import requests
import json

def process_images(directory):
    url = "https://ilocr.iiit.ac.in/ocr/google"
    payload = {'language': 'en', 'token': '747b7b2b-8740-488f-a942-b0ddd7fda7af'}
    results = {}

    for filename in os.listdir(directory):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(directory, filename)
            
            with open(image_path, 'rb') as image_file:
                files = [('image', (filename, image_file, 'image/jpeg'))]
                response = requests.post(url, data=payload, files=files)
                
                print(response)
                
                if response.status_code == 200:
                    ocr_result = response.json()
                    results[filename] = ocr_result
                else:
                    print(f"Error processing {filename}: {response.status_code}")

    return results

def save_results(results, output_file):
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

# Main execution
if __name__ == "__main__":
    image_directory = "./images/posters/Test"  # Replace with your image directory path
    output_file = "ocr_results.json"

    results = process_images(image_directory)
    save_results(results, output_file)
    print(f"OCR results have been saved to {output_file}")