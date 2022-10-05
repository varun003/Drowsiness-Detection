import tkinter as tk
from tkinter import *
import cv2
from references import *
from PIL import Image, ImageTk

# Create an instance of TKinter Window or frame
win = Tk()

# Set the size of the window
win.geometry("600x550")

f1 = Frame(win,bg='black').grid()


# Create a Label to capture the Video frames
label =Label(win)
label.grid(row=0, column=1)

b1 = Button(win,text='START',bg='green',command=start) # command = 'text' for connecting button with logic
b2 = Button(win,text='STOP',bg='red')

b1.grid(row=1,column=1,sticky=SW,padx=250,pady=10)
b2.grid(row=1,column=1,sticky=SW,padx=300,pady=10)

cap= cv2.VideoCapture(0)

# Define function to show frame
def show_frames():
   # Get the latest frame and convert into Image
   cv2image= cv2.cvtColor(cap.read()[1],cv2.COLOR_BGR2RGB)
   cv2image = cv2.flip(cv2image,1)
   img = Image.fromarray(cv2image)
   # Convert image to PhotoImage
   imgtk = ImageTk.PhotoImage(image = img)
   label.imgtk = imgtk
   label.configure(image=imgtk)
   # Repeat after an interval to capture continiously
   label.after(20, show_frames)

show_frames()
win.mainloop()