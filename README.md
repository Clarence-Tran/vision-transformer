# Google-ViT for Image Classification Task

## ViT Model

Vision Transformer (ViT) is a Transformer-based deep learning model designed for computer vision tasks. Instead of using convolutional layers like traditional CNNs, ViT splits an image into fixed-size patches, converts them into embeddings, and processes them using the Transformer encoder architecture with *self-attention mechanism*. The model is typically pretrained on large image datasets and then fine-tuned for downstream tasks (such as image classification).

More details: https://huggingface.co/docs/transformers/model_doc/vit

---

## Flower Image Classification Application

This project is a Streamlit-based web application for flower image classification using a fine-tuned Vision Transformer (ViT) model. The application allows users to upload an image of a flower and obtain the predicted flower class from the trained model.

---

## Project Structure


```text
├── app
│   ├── app.py
│   ├── requirements.txt
│   └── image
│       └── roses.jpg
│
├── ViT_flower_classification.ipynb
├── DETR model.ipynb
├── [My Blog] - step-by-step ViT.ipynb
├── .gitignore
└── README.md
```

## Quickstart

Navigate to the **app directory** and install dependencies.

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```
After running the command, the Streamlit interface will open in your browser where you can upload a flower image and obtain classification results.

## Online Demo
Access the live demo here: https://huytq-flower-cls.streamlit.app/

**Author**: **Quang-Huy Tran, Student at HCMC University of Technology, VNU-HCM**
