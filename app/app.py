import streamlit as st
from transformers import pipeline
from PIL import Image

@st.cache_resource
def load_model():
    return pipeline("image-classification", model="huytqvn/vit-demo")

classifier = load_model()
en2vn = {'daisy': 'cúc họa mi', 'dandelion': 'bồ công anh', 'roses': 'hoa hồng', 'sunflowers':'hướng dương', 'tulips' : 'tuy líp'}

def main():
    st.set_page_config(page_title="Flower Classification")
    st.title("Tính năng phân loại hoa")
    st.caption("Tác giả: Quang Huy, ĐH Bách Khoa TP.HCM")
    st.caption("Model: vit-base-patch16-224-in21k | Dataset: Flower Dataset")
    st.caption("Các loại hoa được huấn luyện: cúc họa mi, bồ công anh, hoa hồng, hoa hướng dương và hoa tuy líp.")

    option_for_user = st.selectbox("Chọn cách thực hiện.", ("Tải lên một ảnh", "Chạy ảnh mẫu"))
    if option_for_user == 'Tải lên một ảnh':
        # convert file to PIL.Image
        file = st.file_uploader("Please upload an image", type=["png", "jpg", "jpeg"])
        if file is not None:
            img = Image.open(file)
            img.thumbnail((300, 300)) 
            result = classifier(img)[0]
            label, score = result['label'], result['score']*100
            st.image(img)
            st.success(f"Loại hoa: {en2vn[label]}. Độ chính xác {score:.2f}%")

    elif option_for_user == "Chạy ảnh mẫu":
        image = Image.open('./image/roses.jpg')
        image.thumbnail((300, 300)) 

        result = classifier(image)[0]
        label, score = result['label'], result['score']*100
        st.image(image)
        st.success(f"Loại hoa: {en2vn[label]}. Độ chính xác {score:.2f}%")


if __name__ == '__main__':
    main()