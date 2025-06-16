import tkinter as tk
import numpy as np
from PIL import Image, ImageOps, ImageGrab

# --- Dummy neural net weights (replace with real trained weights for real predictions) ---
np.random.seed(0)
W1 = np.random.randn(784, 128) * 0.01
b1 = np.zeros((1, 128))
W2 = np.random.randn(128, 10) * 0.01
b2 = np.zeros((1, 10))

# --- Activation functions ---
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# --- Forward pass ---
def predict_digit(img_array):
    z1 = img_array @ W1 + b1
    a1 = relu(z1)
    z2 = a1 @ W2 + b2
    a2 = softmax(z2)
    return np.argmax(a2), np.max(a2)

# --- GUI ---
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Draw a digit (0-9)")
        self.canvas = tk.Canvas(self, width=280, height=280, bg='black')
        self.canvas.pack()
        self.button_predict = tk.Button(self, text="Predict", command=self.on_predict)
        self.button_predict.pack()
        self.button_clear = tk.Button(self, text="Clear", command=self.on_clear)
        self.button_clear.pack()
        self.label_result = tk.Label(self, text="", font=("Helvetica", 20))
        self.label_result.pack()
        self.canvas.bind("<B1-Motion>", self.paint)
        self.last_x, self.last_y = None, None

    def paint(self, event):
        x, y = event.x, event.y
        if self.last_x and self.last_y:
            self.canvas.create_line(self.last_x, self.last_y, x, y, width=15, fill='white', capstyle=tk.ROUND, smooth=True)
        self.last_x, self.last_y = x, y

    def on_clear(self):
        self.canvas.delete("all")
        self.last_x, self.last_y = None, None
        self.label_result.config(text="")

    def on_predict(self):
        # Grab the canvas area from screen
        self.canvas.update()
        x = self.canvas.winfo_rootx()
        y = self.canvas.winfo_rooty()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()

        img = ImageGrab.grab().crop((x, y, x1, y1))
        img = img.convert('L')
        img = ImageOps.invert(img)
        img = img.resize((28, 28))
        img_array = np.array(img).astype(np.float32)
        img_array /= 255.0
        img_array = img_array.flatten().reshape(1, -1)

        pred_digit, confidence = predict_digit(img_array)
        self.label_result.config(text=f"Predicted: {pred_digit} (confidence: {confidence:.2f})")

# --- Run app ---
if __name__ == "__main__":
    app = App()
    app.mainloop()
