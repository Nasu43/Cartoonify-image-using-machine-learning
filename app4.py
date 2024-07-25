import os
import numpy as np
import tensorflow as tf
import streamlit as st
from PIL import Image, ImageEnhance
from io import BytesIO

# Load image
def li(img_bytes):
    img = Image.open(img_bytes)
    img = np.array(img)
    img = img.astype(np.float32) / 127.5 - 1
    img = np.expand_dims(img, 0)
    img = tf.convert_to_tensor(img)
    return img

# Preprocess image
def pi(img, td=224):
    shp = tf.cast(tf.shape(img)[1:-1], tf.float32)
    sd = min(shp)
    scl = td / sd
    nhp = tf.cast(shp * scl, tf.int32)
    img = tf.image.resize(img, nhp)
    img = tf.image.resize_with_crop_or_pad(img, td, td)
    return img

def cartoon(img_bytes):
    try:
        # Loading image
        si = li(img_bytes)
        psi = pi(si, td=512)

        # Model dataflow
        m = '1.tflite'
        if not os.path.exists(m):
            raise FileNotFoundError(f"Model file {m} does not exist.")
        i = tf.lite.Interpreter(model_path=m)
        ind = i.get_input_details()
        i.allocate_tensors()
        i.set_tensor(ind[0]['index'], psi)
        i.invoke()
        r = i.tensor(i.get_output_details()[0]['index'])()

        # Post process the model output
        o = (np.squeeze(r) + 1.0) * 127.5
        o = np.clip(o, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        o = Image.fromarray(o)

        return o

    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

# Apply brightness and sharpness
def adjust_image(img, brightness=1.0, sharpness=1.0):
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(img)
    img = enhancer.enhance(brightness)
    
    # Adjust sharpness
    enhancer = ImageEnhance.Sharpness(img)
    img = enhancer.enhance(sharpness)
    
    return img

# Streamlit app
def main():
    st.title("Image Cartoonification")

    uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Convert to PIL Image
        img = Image.open(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)
        st.write("Processing...")

        # Cartoonify
        cartoonified_image = cartoon(uploaded_file)

        if cartoonified_image is not None:
            # Sliders for adjustments
            brightness = st.slider("Brightness", 0.0, 2.0, 1.0)
            sharpness = st.slider("Sharpness", 0.0, 2.0, 1.0)

            # Adjust the cartoonified image
            adjusted_image = adjust_image(cartoonified_image, brightness, sharpness)

            # Display adjusted image
            st.image(adjusted_image, caption='Adjusted Cartoonified Image', use_column_width=True)

            # Convert image to bytes for download
            buffered = BytesIO()
            adjusted_image.save(buffered, format="PNG")
            img_bytes = buffered.getvalue()

            # Download button
            st.download_button(
                label="Download Cartoonified Image",
                data=img_bytes,
                file_name="cartoonified_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
