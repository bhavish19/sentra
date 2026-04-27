import base64
import io

import tensorflow as tf
from PIL import Image

class MNIST:

    def __init__(self):
        print("Loading MNIST test data …")
        (_, _), (self.x_test, self.y_test) = tf.keras.datasets.mnist.load_data()

        # Build a mapping: digit (0-9) → sorted list of indices into x_test
        self.digit_index: dict[int, list[int]] = {d: [] for d in range(10)}
        for idx, label in enumerate(self.y_test):
            self.digit_index[int(label)].append(idx)
        for d in self.digit_index:
            self.digit_index[d].sort()

        print(f"MNIST loaded. Test-set size: {len(self.x_test)} images.")
    
    def array_to_b64_png(self,arr) -> str:
        """Convert a 28×28 numpy array to a base64-encoded PNG data-URI string."""
        img = Image.fromarray(arr.astype("uint8"), "L")
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{encoded}"

    def getJSONObjectForTestImage(self,index:int)->object:
        return {
            "index": index,
            "label": int(self.y_test[index]),
            "image_b64": self.array_to_b64_png(self.x_test[index]),
        }
    
    def getJSONObjectForDigit(self,digit:int)->object:
        results = []
        for idx in self.digit_index[digit]:
            results.append({
                "index": idx,
                "image_b64": self.array_to_b64_png(self.x_test[idx]),
            })

        return results

