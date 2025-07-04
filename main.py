import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm  # Use standard tqdm for .py files
from torchvision import datasets, transforms

mnist_train = datasets.MNIST(root='./datasets', train=True, transform=transforms.ToTensor(), download=True)
mnist_test = datasets.MNIST(root='./datasets', train=False, transform=transforms.ToTensor(), download=True)
train_loader = torch.utils.data.DataLoader(mnist_train, batch_size=100, shuffle=True)
test_loader = torch.utils.data.DataLoader(mnist_test, batch_size=100, shuffle=False)

W = torch.randn(784, 10)/np.sqrt(784)
W.requires_grad_()  # Use requires_grad_() method with underscore
b = torch.zeros(10, requires_grad=True)

optimizer = torch.optim.SGD([W,b], lr=0.1)

for images, labels in tqdm(train_loader):
    optimizer.zero_grad()

    x = images.view(-1, 784)
    y = torch.matmul(x, W) + b
    cross_entropy = F.cross_entropy(y, labels)

    cross_entropy.backward()
    optimizer.step()

correct = 0
total = len(mnist_test)

with torch.no_grad():
    for images, labels in tqdm(test_loader):
        x = images.view(-1, 784)
        y = torch.matmul(x,W) + b
        cross_entropy = F.cross_entropy(y, labels)

        predictions = torch.argmax(y, dim=1)
        correct += torch.sum(predictions==labels).float()

torch.save({'W':W, 'b': b}, './models/logreg_model.pth')

import tkinter as tk
from PIL import Image, ImageDraw, ImageOps

checkpoint = torch.load('./models/logreg_model.pth')
W = checkpoint['W']
b = checkpoint['b']

class App:
    def __init__(self, root):
        self.canvas_width = 280
        self.canvas_height = 280
        self.root = root
        self.canvas = tk.Canvas(root, width=self.canvas_width, height=self.canvas_height, bg='black')
        self.canvas.pack()

        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)

        self.canvas.bind("<B1-Motion>", self.paint)

        btn_predict = tk.Button(root, text='Predict', command=self.predict_digit)
        btn_predict.pack()

        btn_clear = tk.Button(root, text='Clear', command=self.clear_canvas)
        btn_clear.pack()

    def paint(self, event):
        x1, y1 = (event.x - 10), (event.y - 10)
        x2, y2 = (event.x + 10), (event.y + 10) 
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw .ellipse([x1,y1,x2,y2], fill=255)

    def clear_canvas(self):    
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height),0)
        self.draw = ImageDraw.Draw(self.image)

    def predict_digit(self):
        img_resized = ImageOps.fit(self.image, (28,28), centering=(0.5,0.5))
        #img_inverted = ImageOps.invert(img_resized)
        img_tensor = torch.tensor(list(img_resized.getdata()), dtype=torch.float32).reshape(1,784)/255.0

        with torch.no_grad():
            logits = torch.matmul(img_tensor, W) + b
            predicted_digit = torch.argmax(logits, dim=1).item()
            print("Predicted digit: ", predicted_digit)


root = tk.Tk()
root.title("Draw a digit")
app = App(root)
root.mainloop()
