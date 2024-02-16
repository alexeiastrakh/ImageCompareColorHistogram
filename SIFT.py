import requests
import numpy as np
import cv2 as cv
import json

THRESHOLD = 10
CSV_ROW_LENGTH = 22
ROOT_RANK = 0
TRAIN_DATA = "train_task1.json"
TEST_DATA = "test_task1.json"
VALIDATION_DATA = "val_task1.json"
CSV_DST_FILE = "result.csv"


def file_to_string(file_name):
    with open(file_name, 'r') as file:
        return file.read()


def save_file(file_name, data):
    with open(file_name, 'w') as file:
        file.write(data)


def download_image(url):
    response = requests.get(url)
    if response.status_code != 200:
        raise RuntimeError(f"Image {url} could not be loaded!")
    buffer = np.frombuffer(response.content, dtype=np.uint8)
    image = cv.imdecode(buffer, cv.IMREAD_COLOR)
    return image


def compare_images(url1, url2):
    image1 = download_image(url1)
    image2 = download_image(url2)

    gray1 = cv.cvtColor(image1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()

    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    bf = cv.BFMatcher()

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    return 1 if len(good_matches) > THRESHOLD else 0


def load_data(file_name):
    with open(file_name, 'r') as file:
        return json.load(file)["data"]["results"]


def process_image_pair(image_pair):
    representative_data = image_pair["representativeData"]
    first_image_url = representative_data["image1"]["imageUrl"]
    second_image_url = representative_data["image2"]["imageUrl"]

    return compare_images(first_image_url, second_image_url)


def generate_csv(results):
    local_answer = "taskId,answer\n"
    for image_pair in results:
        local_answer += f"{image_pair['taskId']},{int(process_image_pair(image_pair))}\n"
    with open(CSV_DST_FILE, 'w') as file:
        file.write(local_answer)


def compare_model():
    results = load_data(VALIDATION_DATA)
    generate_csv(results)


if __name__ == "__main__":
    compare_model()

