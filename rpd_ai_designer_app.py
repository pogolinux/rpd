import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch

st.set_page_config(page_title="RPD AI Designer", layout="centered")

st.title("🦷 AI 기반 RPD 디자인 시뮬레이터")

uploaded_file = st.file_uploader("치아 악궁 이미지를 업로드하세요", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # YOLOv5 모델 로드
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # 모델 경로 필요

    # 이미지 numpy 배열로 변환
    img = np.array(image)

    # 모델 예측
    results = model(img)

    # 결과 이미지 시각화
    results.render()
    st.image(results.ims[0], caption="RPD 디자인 결과", use_column_width=True)
