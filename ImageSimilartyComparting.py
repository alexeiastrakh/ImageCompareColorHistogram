import cv2 as cv
import numpy as np
import json
import requests

TEST_DATA = "val_task1.json"
CSV_DST_FILE = "res2.csv"
def download_image(url):
    response = requests.get(url)
    if response.status_code == 200:
        nparr = np.frombuffer(response.content, np.uint8)
        img = cv.imdecode(nparr, cv.IMREAD_COLOR)
        return img
    else:
        raise Exception(f"Failed to download image from URL: {url}")

def compare_images(base_image, test_image):
    hsv_base = cv.cvtColor(base_image, cv.COLOR_BGR2HSV)
    hsv_test = cv.cvtColor(test_image, cv.COLOR_BGR2HSV)

    h_bins = 50
    s_bins = 60
    histSize = [h_bins, s_bins]
    h_ranges = [0, 180]
    s_ranges = [0, 256]
    ranges = h_ranges + s_ranges
    channels = [0, 1]

    hist_base = cv.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)
    hist_test = cv.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
    cv.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv.NORM_MINMAX)

    compare_method = cv.HISTCMP_CORREL
    similarity = cv.compareHist(hist_base, hist_test, compare_method)
    if similarity > 0.4:
        return 1
    else:
        return 0

def test_model():
    total_recognized_duplicates = 0
    num_processed_images = 0
    with open(TEST_DATA, 'r') as file:
        data = json.load(file)
        results = data["data"]["results"]
        total_images = len(results)
        local_answer = "taskId,answer\n"
        for idx, image_pair in enumerate(results, start=1):
            representative_data = image_pair["representativeData"]
            first_image_url = representative_data["image1"]["imageUrl"]
            second_image_url = representative_data["image2"]["imageUrl"]

            first_image = download_image(first_image_url)
            second_image = download_image(second_image_url)


            is_recognized = compare_images(first_image,second_image)
            total_recognized_duplicates += int(is_recognized)

            num_processed_images += 1
            local_answer += f"{image_pair['taskId']},{int(is_recognized)}\n"
            progress_percentage = (num_processed_images / total_images) * 100
            print(f"Processed {num_processed_images} out of {total_images} images ({progress_percentage:.2f}%).", end="\r")
    with open(CSV_DST_FILE, 'w') as file:
        file.write(local_answer)

if __name__ == "__main__":
    test_model()
