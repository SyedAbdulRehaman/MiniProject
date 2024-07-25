import numpy as np
import cv2
import streamlit as st
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# Colorizer function
def colorizer(img):
    try:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
        
        prototxt = "models/models_colorization_deploy_v2.prototxt"
        model = "models/colorization_release_v2.caffemodel"
        points = "models/pts_in_hull.npy"
        
        print("Loading model and points...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        pts = np.load(points)
        
        class8 = net.getLayerId("class8_ab")
        conv8 = net.getLayerId("conv8_313_rh")
        pts = pts.transpose().reshape(2, 313, 1, 1)
        net.getLayer(class8).blobs = [pts.astype("float32")]
        net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
        
        scaled = img_rgb.astype("float32") / 255.0
        lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
        
        resized = cv2.resize(lab, (224, 224))
        L = cv2.split(resized)[0]
        L -= 50
        
        net.setInput(cv2.dnn.blobFromImage(L))
        ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
        
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
        
        L = cv2.split(lab)[0]
        colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
        
        colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
        colorized = np.clip(colorized, 0, 1)
        colorized = (255 * colorized).astype("uint8")
        
        print("Image colorized successfully.")
        return colorized, img_rgb
    except Exception as e:
        print(f"Error in colorizing image: {e}")
        return None, None

# Function to compute PSNR
def compute_psnr(original, colorized):
    try:
        # Convert images to float32 and normalize to [0, 255]
        original = original.astype(np.float32)
        colorized = colorized.astype(np.float32)
        
        # Compute PSNR with data_range explicitly set to 255 (for 8-bit images)
        psnr = compare_psnr(original, colorized, data_range=255)
        
        return psnr
    except Exception as e:
        print(f"Error computing PSNR: {e}")
        return None

# Main application function
def main():
    st.markdown("""
        <style>
        .main-title {
            color: black;
            font-size: 48px;
            font-weight: bold;
            text-align: center;
        }
        .subtitle {
            color: black;
            font-size: 24px;
            text-align: center;
        }
        .upload-box {
            background-color: yellow;
            padding: 10px;
            border-radius: 10px;
        }
        .image-container {
            text-align: center;
        }
        .image-container img {
            border: 5px solid black;
            border-radius: 10px;
        }
        .stButton>button {
            background-color: black;
            color: yellow;
            font-weight: bold;
        }
        </style>
        """, unsafe_allow_html=True)

    st.markdown('<h1 class="main-title">Colorize your Black and White Image</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">This is an app to colorize your B&W images. Created by Team - 3</p>', unsafe_allow_html=True)
    
    file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "png"])
    
    if file is None:
        st.markdown('<div class="upload-box"><p>You haven\'t uploaded an image file</p></div>', unsafe_allow_html=True)
    else:
        try:
            image = Image.open(file)
            img = np.array(image)
            
            st.markdown('<div class="image-container"><p>Your original image</p></div>', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            
            colorized_img, grayscale_img = colorizer(img)
            
            if colorized_img is not None:
                st.markdown('<div class="image-container"><p>Your colorized image</p></div>', unsafe_allow_html=True)
                # st.image(colorized_img, use_column_width=True)
                
                # Save and display the colorized image
                colorized_img_pil = Image.fromarray(colorized_img)
                colorized_img_pil.save("colorized_image.png")
                st.image("colorized_image.png", use_column_width=True)
                
                # Compute PSNR
                psnr = compute_psnr(grayscale_img, colorized_img)
                
                if psnr is not None:
                    st.markdown(f"<p><b>PSNR:</b> {psnr:.2f}</p>", unsafe_allow_html=True)
                else:
                    st.text("Error computing PSNR.")
            else:
                st.text("Error in colorizing the image.")
        except Exception as e:
            st.text(f"Error processing image: {e}")

# Run the main function
if __name__ == "__main__":
    main()
