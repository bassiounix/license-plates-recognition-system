import cv2
from flask import Flask, render_template, request
from utils import plate_detector
from utils import read_plate_numbers
from utils import annotate_plates_with_bbox
from utils import get_cropped_area_gray


app = Flask(__name__, template_folder='./server/templates', static_url_path='', static_folder='server/images')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/', methods=['POST'])
def predict_img():
    imagefile = request.files['imagefile']
    image_path = "./server/images/" + imagefile.filename
    imagefile.save(image_path)

    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cropped_imgs = get_cropped_area_gray(img_rgb)
    plate_object_detector = plate_detector(img_rgb)
    plate_numbers = read_plate_numbers(cropped_imgs)
    annotated_img = annotate_plates_with_bbox(img_rgb, plate_object_detector, plate_numbers)

    detected_image_path = "./server/images/detected_" + imagefile.filename
    cv2.imwrite(detected_image_path, annotated_img)

    return render_template('index.html', path=f"/detected_{imagefile.filename}")


if __name__ == '__main__':
    app.run()
