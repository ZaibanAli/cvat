import numpy as np
import tensorflow as tf
import cv2
import keras
import math


class ModelLoader:
    def __init__(self, model_path):
        self.chunk_size = 70
        self.input_size = 227
        self.resize = 2
        self.model = keras.models.load_model(model_path)

    def infer(self, buf):
        original_image = cv2.imdecode(np.frombuffer(buf.read(), np.uint8), cv2.IMREAD_COLOR)
        resized_image = cv2.resize(original_image,
                                   (original_image.shape[1] // self.resize, original_image.shape[0] // self.resize))

        cookies = self._get_segments(resized_image, self.chunk_size, self.chunk_size, .4)
        test = [c["crop"] for c in cookies]
        test = np.array(test).reshape((-1, self.input_size, self.input_size, 3)) / 255
        predictions = self.model.predict(test).tolist()
        results = [[p.index(max(p)), max(p)] for p in predictions]
        results = [[i] + cookies[i]["offset"] + r for i, r in enumerate(results)]
        return results

    def _get_segments(self, image, cwidth, cheight, overlap=0.0):
        height, width = image.shape[0], image.shape[1]
        num_crops_w = 1 + math.ceil((width - cwidth) / (cwidth * (1 - overlap)))
        num_crops_h = 1 + math.ceil((height - cheight) / (cheight * (1 - overlap)))

        coords_x1 = []
        coords_x2 = []
        for j in range(num_crops_w):
            if j == 0:
                coords_x1.append(0)
                coords_x2.append(cwidth)
            else:
                coords_x1.append(j * int(cwidth * (1 - overlap)))
                coords_x2.append(j * int(cwidth * (1 - overlap)) + cwidth)

        coords_y1 = []
        coords_y2 = []
        for j in range(num_crops_h):
            if j == 0:
                coords_y1.append(0)
                coords_y2.append(cheight)
            else:
                coords_y1.append(j * int(cheight * (1 - overlap)))
                coords_y2.append(j * int(cheight * (1 - overlap)) + cheight)

        if len(coords_y2) == 0:
            coords_y1.append(0)
            coords_y2.append(cheight)

        if len(coords_x2) == 0:
            coords_x1.append(0)
            coords_x2.append(cwidth)

        if max(coords_x2) - max(coords_x1) != cwidth:
            coords_x2[-1] = max(coords_x1) + cwidth

        if max(coords_y2) - max(coords_y1) != cheight:
            coords_y2[-1] = max(coords_y1) + cheight

        crops = []
        for k in range(num_crops_w):
            for m in range(num_crops_h):
                cc = [coords_x1[k], coords_y1[m], coords_x2[k], coords_y2[m]]
                crops.append(cc)

        crop_out = list()
        cropped = []
        for k, cr in enumerate(crops):
            if cr[2] > width - 1:
                cr[2] = width - 1

            if cr[3] > height - 1:
                cr[3] = height - 1

            crop = image[int(cr[1]):int(cr[3]), int(cr[0]):int(cr[2])]
            padded = []

            if crop.shape[0] < cheight:
                padded = np.ones((cheight, cwidth, 3))
                padded[:crop.shape[0], :crop.shape[1]] = crop[:cheight, :cwidth]

            if crop.shape[1] < cwidth:
                padded = np.ones((cheight, cwidth, 3))
                padded[:crop.shape[0], :crop.shape[1]] = crop[:cheight, :cwidth]

            if crop.shape[0] == cheight and crop.shape[1] == cwidth:
                padded = crop

            crop_out.append({
                "offset": cr,
                "crop": cv2.resize(padded, (self.input_size, self.input_size))
            })
            cropped.append(crop)

        # print("Number of Crops: ", len(crops))
        return crop_out

    def extraction(self, rects, labels):
        final_result = list()

        for rect in rects:
            boxes = [self.resize * r for r in rect[1:5]]  # [x1, y1, x2, y2]
            label = labels.get(rect[5], "unknown")
            confidence = rect[6]
            final_result.append({
                "confidence": str(confidence),
                "label": label,
                "points": boxes,
                "type": "rectangle"
            })

        return final_result
