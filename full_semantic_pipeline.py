# Run OCR on an image

import cv2
import numpy as np
import matplotlib.pyplot as plt
# import pytesseract
# from pytesseract import Output
from PIL import Image
# import pytesseract
import keras_ocr
import os
import pickle as pkl
from llm_interface import LLM_OCR_to_Classes_Interface
import autocorrect
from vild_obj_detec import my_main

pipeline = keras_ocr.pipeline.Pipeline()
i = 0

objects = ["vitamins", "fish oil", "omega-3", "COQ10", "aspirin", "tylenol", "ibuprofen", "advil", "calcium", "probiotics", "protein powder", "shampoo", "conditioner", "toothpaste", "face wash", "body wash", "deodorant", "lotion", "sunscreen", "hand cream", "band-aid", "stomach ache relief", "blueberry extracts", "eye nutrition"]

affinity_matrix_objects = np.load('affinity_matrix_pharma_updated.pkl', allow_pickle=True).mean(axis=0)
print("pharma affinity matrix shape", affinity_matrix_objects.shape)

def get_fake_prob_vector(obj_name):
    prob_vector = np.ones((len(objects),))
    prob_vector[objects.index(obj_name)] += 10
    return prob_vector / np.sum(prob_vector)

def get_text_from_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # gray = cv2.medianBlur(gray, 3)
    image = cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            cv2.THRESH_BINARY,11,2)
    plt.imshow(image, cmap='gray')
    plt.show()

    results = pytesseract.image_to_data(image, output_type=Output.DICT, config='--psm 11')

    for i in range(0, len(results['text'])):
        x = results['left'][i]
        y = results['top'][i]

        w = results['width'][i]
        h = results['height'][i]

        text = results['text'][i]
        conf = int(results['conf'][i])

        if conf > 40:
            text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, text, (x, y - 10), 
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 2)

    plt.imshow(image, cmap='gray')
    plt.show()

def run_object_detection(image):
    # get fake object detections for testing purposes
    # image_path = '/home/ravenhuang/sss/pharmacy/IMG_1424.jpg'  #@param {type:"string"}
    # display_image(image_path, size=display_input_size)

    category_name_string = ';'.join(["vitamins", "fish-oil", "omega-3", "calcium", "probiotics", "protein-powder", "shampoo", "conditioner", "toothpaste", "face-wash", "body-wash", "deodorant", "lotion", "sunscreen", "hand cream", "band-aid", "aspirin", "tylenol", "ibuprofen", "advil", "shaving cream", "nail-polish", "toothpaste", "toothbrush", "dental floss", "disinfection-wipe", "eyedrops", "COQ-10"," stomach pain-relief", "blueberry-extraction"])
    max_boxes_to_draw = 10 #@param {type:"integer"}

    nms_threshold = 0.6 #@param {type:"slider", min:0, max:0.9, step:0.05}
    min_rpn_score_thresh = 0.9  #@param {type:"slider", min:0, max:1, step:0.01}
    min_box_area = 220 #@param {type:"slider", min:0, max:10000, step:1.0}
    params = max_boxes_to_draw, nms_threshold, min_rpn_score_thresh, min_box_area

    outputs = my_main(image, category_name_string, params)
    return outputs
"""     fake_object_detections = [
        {'prob_vector': get_fake_prob_vector('stomach ache relief'), 'pos': np.array([2591, 1400, 3080, 2100])},
        {'prob_vector': get_fake_prob_vector('tylenol'), 'pos': np.array([3025, 1291, 3295, 1855])},
        {'prob_vector': get_fake_prob_vector('eye nutrition'), 'pos': np.array([821, 1407, 1447, 2180])},
    ]
    return fake_object_detections

    # will return detections of the following format
    # detections = [
        # ([class 0 prob, class 1 prob, ...], (x1, y1, x2, y2))...
    # ]
    
    # return detections """

def get_text_from_image_keras_ocr(image, viz=True):
    global i
    i+=1
    images = [image]
    prediction_groups = pipeline.recognize(images, detection_kwargs={'detection_threshold': 0.6})
    pred = prediction_groups[0]

    if viz:
        # draw predictions on image
        annotated_img = images[0].copy()
        for j in range(len(pred)):
            box = pred[j][1]
            cv2.rectangle(annotated_img, (int(box[0][0]), int(box[0][1])), (int(box[2][0]), int(box[2][1])), (0, 255, 0), 2)
            text = pred[j][0]
            cv2.putText(annotated_img, text, (int(box[0][0]), int(box[0][1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 255, 255), 2)

        plt.imsave(f'keras_ocr_{i}.png', annotated_img)
    return pred


def get_center_and_dims(ocr_detection):
    x1, y1, x2, y2 = ocr_detection[1][0][0], ocr_detection[1][0][1], ocr_detection[1][2][0], ocr_detection[1][2][1]
    center = (x1 + x2) / 2
    dims = (x2 - x1, y2 - y1)
    return center, dims


def filter_text_detections(ocr_detections):
    max_ocr_detections = 4
    def sort_key(det):
        _, dims = get_center_and_dims(det)
        return min(dims)
    
    ocr_detections = sorted(ocr_detections, key=sort_key, reverse=True)
    ocr_detections = ocr_detections[:max_ocr_detections]
    return ocr_detections


def refine_object_class_probs_with_ocr(object_detections, ocr_detections):
    # first group the text with the bounded boxes
    for ocr_detection in ocr_detections:
        text_center_xy = (ocr_detection[1][0][0] + ocr_detection[1][2][0]) / 2, (ocr_detection[1][0][1] + ocr_detection[1][2][1]) / 2
        for object_detection in object_detections:
            x_range = object_detection['pos'][0] <= text_center_xy[0] <= object_detection['pos'][2]
            y_range = object_detection['pos'][1] <= text_center_xy[1] <= object_detection['pos'][3]
            if x_range and y_range:
                object_detection['text'] = object_detection.get('text', []) + [ocr_detection]
                break
    
    for object_detection in object_detections:
        object_detection['text'] = filter_text_detections(object_detection['text'])
    ocr_perobject_text = [" ".join([det[0] for det in detection['text']]) for detection in object_detections]

    path = "facebook/opt-125m"
     #"/home/kaushiks/sss/opt-13b/"
    # prompt = "In a household shelf, the {} goes {} the " #Prompt1
    prompt = "The text '{}' would most likely {} " #Prompt2, all previous results were run with prompt2

    NAME = 'affinity_matrix_ocr_to_classes.pkl'
    geo = ["be on the"]
    interface = LLM_OCR_to_Classes_Interface(path, objects, prompt, geo, "causal", name=NAME, ocr_text=ocr_perobject_text)
    affinity_matrix = interface.get_affinity_matrix()

    for i, object_detection in enumerate(object_detections):
        # affinity matrix gives log probabilities, we want to convert to probabilities
        object_detection[0] = object_detection[0] * np.exp(affinity_matrix[i])
    return object_detections

def add_gaussian_to_occupancy_distribution(img_len, mean, sigma=10):
    return np.exp(-np.power(np.arange(img_len) - mean, 2.) / (2 * np.power(sigma, 2.)))

def add_uniform_to_occupancy_distribution(img_len):
    return np.ones((img_len,))

def get_semantic_spatial_distribution(refined_object_detections, target_object, img_len):
    # refined_object_detections is a list of tuples of the form (class_probs, (x1, y1, x2, y2))
    target_object_idx = objects.index(target_object)
    occupancy_distribution = np.zeros((img_len,))

    for obj in refined_object_detections:
        print(obj, affinity_matrix_objects.shape)
        affinity_value = np.mean(affinity_matrix_objects[:, target_object_idx] * obj[0])

        occupancy_distribution += add_gaussian_to_occupancy_distribution(img_len, (obj[1][0] + obj[1][2])/2, sigma=-affinity_value * 0.1 * img_len)
        occupancy_distribution += add_uniform_to_occupancy_distribution(img_len) * 0.1
    return occupancy_distribution


def get_semantic_occupancy_distribution(image, target_object):
    object_detections = run_object_detection(image)
    ocr_detections = get_text_from_image_keras_ocr(image)
    # exit()
    refined_object_affinities = refine_object_class_probs_with_ocr(object_detections, ocr_detections)
    return get_semantic_spatial_distribution(refined_object_affinities, target_object, image.shape[1])

if __name__ == "__main__":
    # for file in os.listdir("test_pharmacy_images"):
    #     if file == ".DS_Store":
    #         continue
    image = cv2.imread('/home/kaushiks/sss/semantic-ss/test_pharmacy_images/IMG_1424.jpg')[..., ::-1]
    # plt.imshow(image)
    # plt.show()
    semantic_occ_dist = get_semantic_occupancy_distribution(image, target_object='ibuprofen')

    # plot the semantic occupancy distribution
    plt.imshow(image)
    plt.plot(np.arange(semantic_occ_dist.shape[0]), (1-semantic_occ_dist/semantic_occ_dist.max())*image.shape[0], color="red", linewidth=3)
    plt.show()