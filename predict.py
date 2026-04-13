import os
import pickle
import tempfile


class DMCModelPortable:
    def __init__(self, weights_path):
        with open(weights_path, "rb") as f:
            self.weights_bytes = f.read()

        self.BIN_CLASS     = 0
        self.GARBAGE_CLASS = 1
        self.BIN_CONF      = 0.35
        self.GARBAGE_CONF  = 0.18
        self.EXPAND_RATIO  = 0.65
        self.INSIDE_THRESH = 0.60

    def _load_detector(self):
        from ultralytics import YOLO

        tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
        tmp.write(self.weights_bytes)
        tmp.close()

        detector = YOLO(tmp.name)
        detector.to("cpu")
        os.unlink(tmp.name)
        return detector

    def predict(self, image_path: str) -> int:
        import cv2
        detector = self._load_detector()

        img = cv2.imread(image_path)
        if img is None:
            return 0
        img_area = img.shape[0] * img.shape[1]

        results = detector(image_path, device="cpu", conf=0.12, verbose=False)[0]

        bins, garbage = [], []

        for box in results.boxes:
            cls  = int(box.cls)
            conf = float(box.conf)
            bbox = box.xyxy[0].tolist()

            if cls == self.BIN_CLASS and conf >= self.BIN_CONF:
                bins.append((bbox, conf))
            elif cls == self.GARBAGE_CLASS and conf >= self.GARBAGE_CONF:
                if self._is_valid_garbage(bbox, img_area):
                    garbage.append((bbox, conf))

        if not bins or not garbage:
            return 0

        for g_bbox, _ in garbage:
            for b_bbox, _ in bins:
                if not self._is_inside_bin(g_bbox, b_bbox) and self._is_near(g_bbox, b_bbox):
                    return 1

        for g_bbox, g_conf in garbage:
            if g_conf >= 0.40:
                for b_bbox, _ in bins:
                    if self._is_near_relaxed(g_bbox, b_bbox):
                        return 1

        return 0

    def _is_valid_garbage(self, bbox, image_area):
        x1, y1, x2, y2 = bbox
        return ((x2-x1)*(y2-y1)) / image_area > 0.002

    def _is_inside_bin(self, g_bbox, b_bbox):
        gx1, gy1, gx2, gy2 = g_bbox
        bx1, by1, bx2, by2 = b_bbox
        ix1, iy1 = max(gx1, bx1), max(gy1, by1)
        ix2, iy2 = min(gx2, bx2), min(gy2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return False
        return ((ix2-ix1)*(iy2-iy1)) / ((gx2-gx1)*(gy2-gy1)) > self.INSIDE_THRESH

    def _is_near(self, g_bbox, b_bbox):
        gx1, gy1, gx2, gy2 = g_bbox
        bx1, by1, bx2, by2 = b_bbox
        pad_x = (bx2-bx1) * self.EXPAND_RATIO
        pad_y = (by2-by1) * self.EXPAND_RATIO
        return (gx1 < bx2+pad_x and gx2 > bx1-pad_x and
                gy1 < by2+pad_y and gy2 > by1-pad_y)

    def _is_near_relaxed(self, g_bbox, b_bbox):
        gx1, gy1, gx2, gy2 = g_bbox
        bx1, by1, bx2, by2 = b_bbox
        pad_x = (bx2-bx1) * 1.0
        pad_y = (by2-by1) * 1.0
        return (gx1 < bx2+pad_x and gx2 > bx1-pad_x and
                gy1 < by2+pad_y and gy2 > by1-pad_y)


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "model.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def predict(model, image_path: str) -> float:
    return float(model.predict(image_path))