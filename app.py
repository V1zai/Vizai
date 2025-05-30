import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pyttsx3

model = YOLO("Vizai/yolov8/weights/best.pt")

engine = pyttsx3.init()

st.set_page_config(page_title="Vizai", layout="centered")
st.title("Vizai")
st.write("Upload an image and detect obstacles in it.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.info("Detecting objects...")
    results = model.predict(img)
    result_img = results[0].plot()

    st.image(result_img, caption="Detection Result", use_column_width=True)

    st.subheader("Detected Objects:")
    detected_any = False
    max_area = 0
    closest_class = None

    for box in results[0].boxes:
        detected_any = True
        xyxy = box.xyxy[0].tolist()
        x1, y1, x2, y2 = map(int, xyxy)
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        cls_name = model.names[cls_id]
        area = (x2 - x1) * (y2 - y1)

        st.write(f"- {cls_name} | [{x1}, {y1}, {x2}, {y2}] | Confidence: {conf:.2f}")

        if area > max_area:
            max_area = area
            closest_class = cls_name

    if detected_any:
        closest_class = closest_class.replace('_', ' ')
        st.success(f"Closest object: {closest_class}")
        engine.say(f"Closest object is {closest_class}")
        engine.runAndWait()
    else:
        st.warning("No objects detected. Try another image.")
