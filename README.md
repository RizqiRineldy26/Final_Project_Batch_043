# SoluSkin – Solusi masalah kulit anda

## Repository Outline

```
├── /Scraping & Data Cleaning - berisi data scraping dan EDA
├── /Data Science and Model Training - berisi notebook data modeling, training, dan inference
├── app_2.py - script deployment Streamlit
├── /model_2_sigmoid.keras - hasil model
```

## Problem Background
```
Industri skincare menyediakan berbagai macam produk dengan berbagai macam kandungan dan fungsi yang berbeda. Bagi Konsumen, banyaknya pilihan produk dapat membuat proses menentukan produk yang sesuai menjadi lebih sulit. American Academy of Dermatology (AAD) menyatakan bahwa banyaknya produk acne yang tersedia secara online maupun di toko dapat membuat seseorang kesulitan dalam menentukan produk yang sesuai. AAD juga menjelaskan bahwa pemilihan treatment perlu mempertimbangkan jenis masalah kulit dan bahan aktif yang digunakan.

Project ini untuk mengidentifikasi kondisi permasalahan pada kulit wajah, sehingga dapat menentukan treatment yang tepat sesuai dengan kondisi pada kulit wajah. Proses ini dilakukan dengan menentukan klasifikasi penyebab masalah pada kulit, memprediksi masalah tersebut, serta memberikan rekomendasi produk penanganan yang tepat untuk mengatasi permasalahan kulit tersebut

```

## Project Output
```
Sebuah app yang di-deploy ke Streamlit yang dapat mendeteksi permasalahan kulit pada wajah dan memberikan rekomendasi ingredients dan rekomendasi produk dari journal dan product database. 
```

## Data
```
- Dataset 1: Skin Issues Dataset — Ahmed Ismail, Kaggle.
- Dataset 2: Facial Skin (Acne, Pigmentation, Pores, Wrinkles) — Shijo John, Kaggle. 
- Scraping Dataset from INKEEDecoder website

```

## Method
```
1. Dataset Gathering and Data Scraping
Mengumpulkan dataset untuk computer vision dari beberapa sumber serta melakukan data scraping dan research journal dengan klasifikasi problem: Acne, Blackheads, Dark Spots, Pores, dan Wrinkles

2. EDA and PreProcessing
Melakukan PreProcessing dengan mengkurasi semua Dataset gambar dan melakukan resizing 150 X 150 serta melakukan Exploratory Data Analysis pada dataset yang sudah dikumpulkan

3. Training & Evaluation
Training dilakukan dengan metode CNN (Computer Vision) beserta Transfer Learning Model EfficientB0 dengan Training 80 Epoch untuk mendapatkan hasil model terbaik.

4. Deployment to Streamlit
Menghubungkan Data dan Jurnal dari Database dengan hasil model untuk memberikan rekomendasi ingredient dan produk menggunakan LLM, lalu melakukan deploy ke Streamlit.
```

## Stacks
```
- streamlit
- numpy
- pandas
- matplotlib
- seaborn
- plotly
- tensorflow
- pillow
- google-genai
- pypdf
- nltk
- chromadb
- langchain-core
- langchain-community
- langchain-google-genai
- langchain-chroma
- langchain-text-splitters
- supabase
```

## Reference
- [Dataset 1: Skin Issues Dataset — Ahmed Ismail, Kaggle.](https://www.kaggle.com/datasets/ahmedismaiil/skin-issues-version-2-dataset-balanced)
- [Dataset 2: Facial Skin (Acne, Pigmentation, Pores, Wrinkles) — Shijo John, Kaggle.](https://www.kaggle.com/datasets/shijo96john/facial-skin-acne-pigmentation-pores-wrinkles)
- [INKEEDecoder](https://inkeedecoder.com/)

---
