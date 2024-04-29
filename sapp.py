import streamlit as st
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('NaturalModel.keras')

def preprocess_data(image_data):
    img = image.load_img(image_data, target_size=(150,150,3))
    img_array = image.img_to_array(img)
    img_array = img_array/255
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def main():
    st.title("Natural Scene Prediction")
    st.write('This app can predict about six different types of natural scene images')
    st.image('th.jpeg', width = 700)
    #caption='Enlarged Image'
    st.write("Introducing an AI adept at discerning the diverse landscapes of forests, glaciers, seas, mountains, streets and buildings. Leveraging advanced machine learning algorithms, it analyzes spatial patterns and features to accurately predict the classification of natural and artificial environments. Trained on vast datasets encompassing varied terrain and structures, the AI boasts robustness and versatility. Whether identifying lush canopies of trees, imposing ice peaks, expansive bodies of water, rugged mountain ranges, or urban architecture, its precision is unmatched. With applications ranging from environmental monitoring to urban planning, our model is poised to revolutionize landscape analysis with unparalleled accuracy and efficiency.")
    st.divider()

    # User Prompt
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        st.image(uploaded_file, caption='Uploaded Image.', use_column_width=True)

        if st.button('Predict'):
            image = preprocess_data(uploaded_file)
            prediction = model.predict(image)
            predicted_class = np.argmax(prediction)
     
            if predicted_class == 0:
                st.success('The Image is Building')

            elif predicted_class == 1:
                st.success('The Image is Forest')
            
            elif predicted_class == 2:
                st.success('The Image is Glacier')
            
            elif predicted_class == 3:
                st.success('The Image is Mountain')

            elif predicted_class == 4:
                st.success('The Image is Sea')

            elif predicted_class == 5:
                st.success('The Image is Street')
            else:
                st.error('Upload Right Image')
    else:
        st.warning('Please upload an image first.')

if __name__ == "__main__":
    main()