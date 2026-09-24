# Judul Project
SoluSkin : Skin Problem Classification & Product Recomendation 

## Repository Outline
1. Data Science and Model Training
    - Computer Vision.ipynb - Source Code Model
    - Inference - Sigmoid.ipynb - Source Code Inference

2. Scrapping and Data Cleaning
    - Scrapping Data
        1. scraping anti-acne.ipynb - Scrapping problem acne
        2. scraping comedo.ipynb - Scrapping problem blackheads
        3. scraping dark spot .ipynb - Scrapping problem dark spots
        4. scraping pores.ipynb - Scrapping problem pores
        5. scraping wrinkle.ipynb - Scrapping problem wrinkles
    - Skincare Dataset
        1. Anti-Acne Product.csv - Data produk untuk problem acne
        2. Comedo Product.csv - Data produk untuk problem blackheads
        3. Dark-Spots Product.csv - Data produk untuk problem dark spots
        4. pores_data.csv - Data produk untuk problem pores
        5. wrinkle.csv - Data produk untuk problem wrinkles
    - raw_data.csv - Data awal setelah scrapping
    - cleaned_data.csv - Data setelah dilakukan cleaning
    - data.ipynb - Source Code proses data cleaning

3. chroma_db_gemini_3072
    - chroma.sqlite3 - File Vector pada Jurnal

4. description.md - Rincian penjelasan Project
5. Batch_043_Logo.jpeg - Project Logo
6. app_2.py - file berisi code python untuk menggabungkan script eda.py dan prediction_test_2.py untuk diproses ke Streamlit
7. eda.py - Exploratory Data Analyst
8. jurnal_skincare.pdf - Rekomendasi Jurnal untuk diintegrasikan ke LLM
9. model_2_sigmoid.keras - File model CNN
10. prediction_test_2.py - file script python inference
11. requirements.txt - File berisi modul yang digunakan

## Problem Background
Banyak pilihan produk skincare dengan kandungan dan fungsi yang beragam membuat konsumen kesulitan menentukan produk yang sesuai sehingga kerap kali menyesal salah beli produk atau salah rekomendasi dari SPG. American Academy of Dermatology (AAD) menegaskan bahwa pemilihan treatment perlu mempertimbangkan jenis masalah kulit dan bahan aktif yang digunakan.

Maka dari itu SoluSkin hadir untuk membantu SPG merekomendasikan produk yang cocok agar menekan jumlah customer dissatisfaction.


## Project Output
- Database Supabase
- Exploratory Data Analyst
- Machine Learning Model dengan CNN
- LLM untuk rekomendasi Bahan dan Produk
- Deployment ke dalam Streamlit

## Data
Sumber Dataset 1 : https://inkeedecoder.com/
Rician Data :
    - 4 Kolom
    - 1945 Baris

Sumber Dataset 2 : https://www.kaggle.com/datasets/ahmedismaiil/skin-issues-version-2-dataset-balanced
Rincian Data :
    - 5 Klasifikasi gambar


## Method
Project ini untuk mengidentifikasi kondisi permasalahan pada kulit wajah, sehingga dapat menentukan treatment yang tepat sesuai dengan kondisi pada kulit wajah. Proses ini dilakukan dengan menentukan klasifikasi penyebab masalah pada kulit, memprediksi masalah tersebut menggunakan Machine Learning model CNN, serta memberikan rekomendasi produk penanganan yang tepat untuk mengatasi permasalahan kulit tersebut menggunakan LLM

## Stacks
Bahasa Pemrograman : Python & SQL Query
Tools : Python, Tensorflow, Supabase, LLM, Streamlit

## Reference
Dataset : 
    - https://inkeedecoder.com/
    - https://www.kaggle.com/datasets/ahmedismaiil/skin-issues-version-2-dataset-balanced
    - https://www.kaggle.com/datasets/shijo96john/facial-skin-acne-pigmentation-pores-wrinkles
    - Google / Pinterest

Deployment : https://soluskin.streamlit.app/



**Referensi tambahan:**
- [Basic Writing and Syntax on Markdown](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
- [Contoh readme](https://github.com/fahmimnalfrzki/Swift-XRT-Automation)
- [Another example](https://github.com/sanggusti/final_bangkit) (**Must read**)
- [Additional reference](https://www.freecodecamp.org/news/how-to-write-a-good-readme-file/)