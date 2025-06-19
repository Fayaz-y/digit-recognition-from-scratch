import tkinter as tk
from PIL import Image, ImageDraw, ImageGrab, ImageOps
import numpy as np

def relu(x): return np.maximum(0, x)
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def crop_bounding_box(img_array):
    rows = np.any(img_array, axis=1)
    cols = np.any(img_array, axis=0)
    if not rows.any() or not cols.any():
        return img_array  # Empty canvas, skip crop
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return img_array[rmin:rmax+1, cmin:cmax+1]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Digit Recognizer")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='white')
        self.canvas.pack()
        self.label_result = tk.Label(self, text="Draw a digit", font=("Arial", 20))
        self.label_result.pack()

        self.button_predict = tk.Button(self, text="Predict", command=self.on_predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.clear_canvas)
        self.button_clear.pack()

        self.canvas.bind("<B1-Motion>", self.draw)

        
        self.last_x, self.last_y = None, None


        weights = np.load("trained_weights.npz")
        self.W1 = weights['W1']
        self.b1 = weights['b1']
        self.W2 = weights['W2']
        self.b2 = weights['b2']

    def draw(self, event):
        if self.last_x is not None and self.last_y is not None:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, width=10, fill='black', capstyle=tk.ROUND)
        self.last_x, self.last_y = event.x, event.y

    def clear_canvas(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
        self.label_result.config(text="Draw a digit")

    def on_predict(self):
        self.canvas.update()
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1)).convert('L')
        img = ImageOps.invert(img)
        img_array = np.array(img)

        img_array = crop_bounding_box(img_array)

        img = Image.fromarray(img_array).resize((28, 28))
        img_array = np.array(img).astype(np.float32) / 255.0

        img_array = img_array.flatten().reshape(1, -1)

    
        digit, conf = self.predict_digit(img_array)
        self.label_result.config(text=f"Predicted: {digit} (confidence: {conf:.2f})")

    def predict_digit(self, img_array):
        z1 = img_array @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = softmax(z2)
        digit = np.argmax(a2)
        conf = np.max(a2)
        return digit, conf

if __name__ == "__main__":
    app = App()
    app.mainloop()
