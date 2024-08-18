import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from fast_plate_ocr import ONNXPlateRecognizer


license_plate_detector = YOLO('license_plate_detector.pt')
m = ONNXPlateRecognizer('argentinian-plates-cnn-synth-model')


def plate_detector(img):
    return license_plate_detector(img)


def get_cropped_area_gray(img_rgb):
    license_plates = plate_detector(img_rgb)[0]
    ret_images = []

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate
        license_plate_crop = img_rgb[int(y1):int(y2), int(x1): int(x2), :]
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        # _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # binarization
        ret_images.append(license_plate_crop_gray)
    
    return ret_images


def read_plate_numbers(cropped_imgs):
    return m.run(cropped_imgs)


def annotate_plates_with_bbox(img_rgb, plate_img_detection, plate_numbers):
    img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB)

    for r in plate_img_detection:
        annotator = Annotator(img_rgb)
        boxes = r.boxes
        for i, box in enumerate(boxes):
            b = box.xyxy[0]  # get box coordinates in (left, top, right, bottom) format
            annotator.box_label(b, plate_numbers[i])

        return annotator.result()
