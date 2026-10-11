import base64
import io
import unittest

import numpy as np
from fastapi import HTTPException
from PIL import Image

from app import decode_image, decode_predictions, iou_single, letterbox, nms, MODEL_CLASSES, check_model_request


class OnnxLogicTests(unittest.TestCase):
    def test_model_has_correct_class(self):
        self.assertEqual(MODEL_CLASSES, {0: "solar-pv-panel"})

    def test_decode_image(self):
        img = Image.new('RGB', (50, 25), 'blue')
        buffer = io.BytesIO()
        img.save(buffer, 'JPEG')
        encoded = base64.b64encode(buffer.getvalue()).decode()
        self.assertEqual(decode_image(encoded).size, (50, 25))
        self.assertEqual(decode_image('data:image/jpeg;base64,' + encoded).size, (50, 25))

    def test_invalid_image(self):
        with self.assertRaises(HTTPException) as context:
            decode_image('%%%')
        self.assertEqual(context.exception.status_code, 400)

    def test_letterbox_and_coordinates(self):
        img = Image.new('RGB', (320, 160))
        tensor, gain, left, top = letterbox(img)
        self.assertEqual(tensor.shape, (1, 3, 640, 640))
        self.assertAlmostEqual(gain, 2.0)
        self.assertEqual((left, top), (0, 160))
        # panel original bbox: x=100..200, y=40..100 -> ONNX xywh: x=300, y=460, w=200, h=120
        prediction = np.zeros((1, 5, 2), dtype=np.float32)
        prediction[:, :, 0] = [300, 300, 200, 120, 0.92]
        prediction[:, :, 1] = [300, 300, 200, 120, 0.20]
        found = decode_predictions(prediction, (320, 160), gain, left, top, 0.25)
        self.assertEqual(len(found), 1)
        self.assertAlmostEqual(found[0]['x'], 100)
        self.assertAlmostEqual(found[0]['y'], 40)
        self.assertAlmostEqual(found[0]['w'], 100)
        self.assertAlmostEqual(found[0]['h'], 60)
        self.assertEqual(found[0]['class_name'], 'solar-pv-panel')

    def test_non_max_suppression(self):
        boxes = np.array([[10, 10, 20, 20], [11, 11, 21, 21], [100, 100, 110, 110]], dtype=np.float32)
        scores = np.array([0.9, 0.8, 0.7])
        self.assertEqual(nms(boxes, scores), [0, 2])

    def test_rejects_different_model(self):
        with self.assertRaises(HTTPException) as context:
            check_model_request('another-model.pt')
        self.assertEqual(context.exception.status_code, 409)


if __name__ == '__main__':
    unittest.main()
