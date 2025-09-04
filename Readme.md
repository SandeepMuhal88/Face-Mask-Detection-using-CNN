# Face Mask Detection Model
That poroject is base on CNN and this is most effective model for face mask detection.

### 🔹 Step 1: Haar Cascade load karna
```python
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
```

OpenCV me Haar Cascade ek pre-trained model hota hai jo faces detect karta hai.

"haarcascade_frontalface_default.xml" ek XML file hai jisme trained face features store hote hain.

face_cascade ab ek face detector ban gaya.

### 🔹 Step 2: Webcam start karna
```python
cap = cv2.VideoCapture(0)
```


0 ka matlab hai laptop ka default webcam.

Agar external camera ho to 1, 2, ... use karte hain.

### 🔹 Step 3: Frame by frame video read karna
```python
ret, frame = cap.read()
```

ret = True/False batata hai ki frame successfully read hua ya nahi.

frame = ek single image (numpy array) jo webcam se liya gaya hai.

### 🔹 Step 4: Image ko grayscale me convert karna
```python
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```

Haar cascade grayscale images par hi work karta hai, isliye RGB ko gray me convert kiya.

### 🔹 Step 5: Face detect karna
```python
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
```

detectMultiScale sab faces detect karta hai.

Parameters:

scaleFactor=1.1 → har step pe image thoda zoom out hota hai taaki different size ke faces detect ho.

minNeighbors=4 → jitni baar detection confirm hona chahiye (kam value = zyada false detections).

faces ek list hota hai → har element (x, y, w, h) (rectangle ke coordinates).

### 🔹 Step 6: Har face ko crop karke model me dena
```python
face = frame[y:y+h, x:x+w]        # Crop face region
face = cv2.resize(face, (224, 224))  # Model input size
face = face / 255.0               # Normalize [0,1] range
face = np.expand_dims(face, axis=0) # Shape (1,224,224,3)
```


Cropped face ko resize kiya model ke input size 224x224 par.

Pixel values ko 0–255 se normalize karke 0–1 ki range me convert kiya.

expand_dims use kiya kyunki model ek batch expect karta hai (shape [1, 224, 224, 3]).

### 🔹 Step 7: Prediction karna
```python
pred = model.predict(face)[0][0]
```

Model ek probability deta hai (0–1 ke beech).

[0][0] ka matlab hai batch ke pehle element ka output value lena.

### 🔹 Step 8: Label decide karna
```python
if pred < 0.5:
    label = "No Mask"
    color = (0, 0, 255)   # Red
else:
    label = "Mask"
    color = (0, 255, 0)   # Green
```

Agar prediction < 0.5 → No Mask (red box).

Agar ≥ 0.5 → Mask (green box).

### 🔹 Step 9: Frame par result draw karna
```python
cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
```


putText → Face ke upar "Mask" ya "No Mask" likhta hai.

rectangle → Face ke around box draw karta hai.

### 🔹 Step 10: Show output window
```python
cv2.imshow("Face Mask Detection", frame)
```


Ek new window open hoti hai jisme live camera ka output dikhai deta hai with detection.

### 🔹 Step 11: Quit condition
```python
if cv2.waitKey(1) & 0xFF == ord('q'):
    break
```


Agar user q press kare toh loop break ho jaata hai aur webcam stop.

###  🔹 Step 12: Cleanup
```python
cap.release()
cv2.destroyAllWindows()
```


cap.release() webcam ko free kar deta hai.

cv2.destroyAllWindows() sab OpenCV windows close kar deta hai.


*** Thanks for using this project ***
