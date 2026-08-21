#  SMS Spam Classification

## 📌 เกี่ยวกับโปรเจกต์

โปรเจกต์นี้เป็นการพัฒนาระบบจำแนกข้อความ SMS ว่าเป็น **Spam (ขยะ)** หรือ **Ham (ปกติ)** โดยใช้เทคนิค Machine Learning และ Natural Language Processing (NLP)

โปรเจกต์นี้จัดทำขึ้นเพื่อเป็น Mini Project ในรายวิชา [ใส่ชื่อวิชา] ภาคการศึกษา [ใส่ภาคการศึกษา] ปีการศึกษา 2569

---

## 👤 ข้อมูลผู้จัดทำ

- **ชื่อ-นามสกุล:** [วชิรวิทย์ พรสวาท]
- **รหัสนักศึกษา:** [664245032]
- **อีเมล:** [664245032@webmail.npru.ac.th]
- **GitHub:** [ใส่ลิงก์ GitHub ของคุณ]

---

## 🎯 วัตถุประสงค์

1. เพื่อศึกษาและประยุกต์ใช้ Machine Learning ในการจำแนกข้อความ
2. เพื่อเปรียบเทียบประสิทธิภาพของอัลกอริทึมต่างๆ (Naive Bayes, Logistic Regression, SVM)
3. เพื่อพัฒนา Web Application ด้วย Streamlit ให้ผู้ใช้สามารถทดสอบโมเดลได้จริง

---

## 📊 Dataset

- **ชื่อ Dataset:** SMS Spam Collection Dataset
- **แหล่งที่มา:** [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) / UCI Machine Learning Repository
- **จำนวนข้อมูล:** 5,572 ข้อความ
- **จำนวน Features:** 2 คอลัมน์ (label, message)
- **Class Distribution:**
  - Ham (ปกติ): 4,825 ข้อความ (~86.6%)
  - Spam (ขยะ): 747 ข้อความ (~13.4%)

---

## 🛠️ เทคโนโลยีและ Libraries ที่ใช้

- **Python 3.x**
- **Pandas** - สำหรับจัดการข้อมูล
- **NumPy** - สำหรับการคำนวณ
- **Scikit-learn** - สำหรับ Machine Learning
- **NLTK** - สำหรับ Natural Language Processing
- **Matplotlib & Seaborn** - สำหรับ Visualization
- **Streamlit** - สำหรับสร้าง Web Application
- **Joblib** - สำหรับบันทึกและโหลดโมเดล

---

##  โครงสร้างไฟล์
sms-spam-classification/
│
├── spam.csv # Dataset ต้นฉบับ
├── requirements.txt # รายชื่อ Python libraries
── README.md # ไฟล์นี้
── app.py # Streamlit Web Application
├── tfidf_vectorizer.pkl # TF-IDF Vectorizer ที่ train แล้ว
├── best_spam_model.pkl # โมเดล SVM ที่ดีที่สุด
├── notebooks/ # Jupyter Notebooks (ถ้ามี)
│ └── spam_classification.ipynb
└── presentation/ # ไฟล์นำเสนอ
└── presentation.pptx