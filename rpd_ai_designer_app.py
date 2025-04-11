import streamlit as st
import cv2
from PIL import Image
import numpy as np
import torch
import urllib.request
import os

st.set_page_config(page_title="RPD AI Designer", layout="centered")
st.title("🦷 AI 기반 RPD 디자인 시뮬레이터")

MODEL_URL = "https://github.com/ultralytics/yolov5/releases/download/v6.0/yolov5s.pt"
MODEL_PATH = "yolov5s.pt"

# 모델 다운로드 (처음 한 번만)
if not os.path.exists(MODEL_PATH):
    with st.spinner("모델 다운로드 중입니다..."):
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        st.success("모델 다운로드 완료!")

# 모델 불러오기
model = torch.hub.load("ultralytics/yolov5", "custom", path=MODEL_PATH)

uploaded_file = st.file_uploader("치아 악궁 이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="업로드된 이미지", use_column_width=True)

    # numpy 배열로 변환
    img_array = np.array(image)

    # 객체 인식 수행
    results = model(img_array)
    results.render()

    # 결과 이미지 표시
    st.image(results.ims[0], caption="인식된 결과", use_column_width=True)
