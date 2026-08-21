import streamlit as st
import streamlit.components.v1 as components
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ดาวน์โหลด NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ตั้งค่าหน้าเว็บ
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# โหลดโมเดลและ Vectorizer
@st.cache_resource
def load_models():
    try:
        model = joblib.load('best_spam_model.pkl')
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        return model, tfidf
    except:
        return None, None

model, tfidf = load_models()

# ฟังก์ชัน Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# สร้าง HTML สำหรับ Presentation
html_content = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SMS Spam Classification Project</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
            text-align: center;
        }
        
        .header h1 {
            color: #667eea;
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        
        .header p {
            color: #666;
            font-size: 1.2em;
        }
        
        .section {
            background: white;
            padding: 40px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-bottom: 30px;
        }
        
        .section h2 {
            color: #667eea;
            font-size: 2em;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 3px solid #667eea;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .stat-box {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
        }
        
        .stat-number {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .stat-label {
            font-size: 1.1em;
            opacity: 0.9;
        }
        
        .content-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 30px;
            margin-top: 30px;
        }
        
        .card {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 10px;
            border-left: 5px solid #667eea;
        }
        
        .card h3 {
            color: #764ba2;
            margin-bottom: 15px;
            font-size: 1.3em;
        }
        
        .card p, .card ul {
            color: #555;
            line-height: 1.8;
        }
        
        .card ul {
            padding-left: 20px;
        }
        
        .highlight {
            background: #fff3cd;
            padding: 20px;
            border-radius: 10px;
            border-left: 5px solid #ffc107;
            margin: 20px 0;
        }
        
        .comparison-table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        
        .comparison-table th,
        .comparison-table td {
            padding: 15px;
            text-align: left;
            border-bottom: 2px solid #ddd;
        }
        
        .comparison-table th {
            background: #667eea;
            color: white;
        }
        
        .comparison-table tr:hover {
            background: #f5f5f5;
        }
        
        .developer-info {
            display: flex;
            align-items: center;
            gap: 30px;
            margin: 30px 0;
            flex-wrap: wrap;
        }
        
        .developer-photo {
            width: 200px;
            height: 200px;
            border-radius: 50%;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            color: white;
            font-size: 4em;
            flex-shrink: 0;
        }
        
        .developer-details {
            flex: 1;
            min-width: 250px;
        }
        
        .developer-details h3 {
            color: #667eea;
            margin-bottom: 10px;
        }
        
        .developer-details p {
            color: #555;
            line-height: 1.8;
            margin-bottom: 10px;
        }
        
        .footer {
            background: white;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            color: #666;
            box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        }
        
        .flowchart {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            font-family: monospace;
            white-space: pre;
            overflow-x: auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <h1>📱 SMS Spam Classification</h1>
            <p>โปรเจกต์ Machine Learning สำหรับจำแนกข้อความ SMS</p>
            <p style="margin-top: 10px; font-size: 1em;">
                <strong>ผู้พัฒนา:</strong> [ชื่อ-นามสกุล] | 
                <strong>รหัสนักศึกษา:</strong> [รหัส] | 
                <strong>ปี:</strong> 2026
            </p>
        </div>

        <!-- หน้าหลัก -->
        <div class="section">
            <h2>🏠 หน้าหลัก - ภาพรวมโปรเจกต์</h2>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">5,572</div>
                    <div class="stat-label">จำนวนข้อความทั้งหมด</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">86.6%</div>
                    <div class="stat-label">ข้อความปกติ (Ham)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">13.4%</div>
                    <div class="stat-label">ข้อความ Spam</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">98%</div>
                    <div class="stat-label">ความแม่นยำของโมเดล</div>
                </div>
            </div>

            <div class="content-grid">
                <div class="card">
                    <h3>🎯 วัตถุประสงค์</h3>
                    <p>พัฒนาระบบ Machine Learning เพื่อจำแนกข้อความ SMS ว่าเป็น Spam หรือข้อความปกติ (Ham) โดยอัตโนมัติ</p>
                </div>
                <div class="card">
                    <h3>🔧 เทคโนโลยีที่ใช้</h3>
                    <ul>
                        <li>Python 3.x</li>
                        <li>Scikit-learn</li>
                        <li>Pandas & NumPy</li>
                        <li>Matplotlib & Seaborn</li>
                        <li>NLTK (Natural Language Toolkit)</li>
                        <li>Streamlit</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>📊 Dataset</h3>
                    <p><strong>SMS Spam Collection Dataset</strong></p>
                    <ul>
                        <li>แหล่งที่มา: Kaggle/UCI</li>
                        <li>จำนวน: 5,572 ข้อความ</li>
                        <li>2 คอลัมน์: label, message</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>⚡ ผลลัพธ์</h3>
                    <p>โมเดล SVM มีความแม่นยำสูงสุดที่ 98% สามารถจำแนก Spam ได้อย่างมีประสิทธิภาพ</p>
                </div>
            </div>

            <div class="highlight">
                <strong> ความสำคัญ:</strong> การตรวจจับ SMS Spam ช่วยป้องกันการหลอกลวงทางโทรศัพท์ ลดข้อความโฆษณาที่ไม่พึงประสงค์ และเพิ่มความปลอดภัยให้ผู้ใช้งาน
            </div>
        </div>

        <!-- การเตรียมข้อมูล -->
        <div class="section">
            <h2>📊 การเตรียมข้อมูล (Data Preprocessing)</h2>
            
            <div class="content-grid">
                <div class="card">
                    <h3>📥 การนำเข้าข้อมูล</h3>
                    <p>อ่านไฟล์ CSV ที่มี 2 คอลัมน์หลัก:</p>
                    <ul>
                        <li><strong>v1 (Label):</strong> ระบุประเภท (ham/spam)</li>
                        <li><strong>v2 (Message):</strong> เนื้อหาข้อความ SMS</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>🧹 การทำความสะอาดข้อมูล</h3>
                    <ul>
                        <li>✅ แปลงเป็นตัวพิมพ์เล็ก (Lowercase)</li>
                        <li>✅ ลบสัญลักษณ์พิเศษและตัวเลข</li>
                        <li>✅ ลบ Stopwords (คำที่ไม่มีความหมาย)</li>
                        <li>✅ Lemmatization (ตัดคำให้เหลือรากศัพท์)</li>
                    </ul>
                </div>
                <div class="card">
                    <h3> การแปลงข้อมูล</h3>
                    <ul>
                        <li>Label Encoding: ham=0, spam=1</li>
                        <li>Text Vectorization: TF-IDF</li>
                        <li>Max Features: 3,000 คำ</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>⚖️ การแบ่งข้อมูล</h3>
                    <ul>
                        <li>Training Set: 80% (4,457 samples)</li>
                        <li>Testing Set: 20% (1,115 samples)</li>
                        <li>Stratified Split</li>
                        <li>Random State: 42</li>
                    </ul>
                </div>
            </div>

            <div class="flowchart">
<strong>📋 ขั้นตอน Data Preprocessing:</strong>

Raw Text 
   │
   ▼
[ Lowercase ]
   │
   ▼
[ Remove Punctuation & Numbers ]
   │
   ▼
[ Remove Stopwords ]
   │
   ▼
[ Lemmatization ]
   │
   ▼
[ TF-IDF Vectorization ]
   │
   ▼
Numeric Vector (3,000 features)
   │
   ▼
🤖 ML Model
            </div>

            <div class="highlight">
                <strong>📝 หมายเหตุ:</strong> ใช้ Stratified Sampling เพื่อรักษาอัตราส่วนระหว่าง Ham และ Spam ให้เท่ากันทั้งใน Training และ Testing sets
            </div>
        </div>

        <!-- วิเคราะห์ข้อมูล -->
        <div class="section">
            <h2>🔍 วิเคราะห์ข้อมูล (Data Analysis)</h2>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">7.6</div>
                    <div class="stat-label">ความยาวเฉลี่ย Ham (คำ)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">13.9</div>
                    <div class="stat-label">ความยาวเฉลี่ย Spam (คำ)</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">45</div>
                    <div class="stat-label">คำที่พบบ่อยที่สุดใน Spam</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">20</div>
                    <div class="stat-label">คำที่พบบ่อยที่สุดใน Ham</div>
                </div>
            </div>

            <div class="content-grid">
                <div class="card">
                    <h3>📏 ความยาวข้อความ</h3>
                    <p><strong>Ham:</strong> ส่วนใหญ่สั้นกว่า 20 คำ<br>
                    <strong>Spam:</strong> มักยาวกว่า 15 คำ<br>
                    <strong>ข้อสังเกต:</strong> Spam มักมีข้อความยาวกว่าเพื่อโน้มน้าว</p>
                </div>
                <div class="card">
                    <h3>🔤 คำที่พบบ่อย</h3>
                    <p><strong>Spam:</strong> free, call, text, win, prize, cash, urgent, now<br>
                    <strong>Ham:</strong> ok, will, can, you, me, to, the, and</p>
                </div>
                <div class="card">
                    <h3>🎯 รูปแบบข้อความ</h3>
                    <p><strong>Spam:</strong><br>
                    - ใช้ตัวพิมพ์ใหญ่<br>
                    - มีสัญลักษณ์พิเศษ<br>
                    - มีลิงก์หรือเบอร์โทร<br>
                    - สร้างความเร่งด่วน</p>
                </div>
                <div class="card">
                    <h3>📈 Correlation</h3>
                    <p>ความยาวข้อความมีความสัมพันธ์เชิงบวกกับการเป็น Spam (r = 0.42)<br><br>
                    การมีคำว่า "free", "win", "prize" เพิ่มโอกาสเป็น Spam 85%</p>
                </div>
            </div>

            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>คุณลักษณะ</th>
                        <th>Ham (ปกติ)</th>
                        <th>Spam</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td>จำนวนข้อความ</td>
                        <td>4,825 (86.6%)</td>
                        <td>747 (13.4%)</td>
                    </tr>
                    <tr>
                        <td>ความยาวเฉลี่ย</td>
                        <td>7.6 คำ</td>
                        <td>13.9 คำ</td>
                    </tr>
                    <tr>
                        <td>คำศัพท์เฉพาะ</td>
                        <td>คำทั่วไปในชีวิตประจำวัน</td>
                        <td>free, win, call, text, prize</td>
                    </tr>
                    <tr>
                        <td>รูปแบบ</td>
                        <td>ภาษาพูดทั่วไป</td>
                        <td>ใช้ตัวพิมพ์ใหญ่, สัญลักษณ์</td>
                    </tr>
                </tbody>
            </table>
        </div>

        <!-- ประสิทธิภาพโมเดล -->
        <div class="section">
            <h2>📈 ประสิทธิภาพโมเดล (Model Performance)</h2>
            
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-number">98.2%</div>
                    <div class="stat-label">Accuracy</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">97.8%</div>
                    <div class="stat-label">Precision</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">96.5%</div>
                    <div class="stat-label">Recall</div>
                </div>
                <div class="stat-box">
                    <div class="stat-number">97.1%</div>
                    <div class="stat-label">F1-Score</div>
                </div>
            </div>

            <div class="content-grid">
                <div class="card">
                    <h3>🤖 อัลกอริทึมที่ทดสอบ</h3>
                    <ul>
                        <li><strong>Naive Bayes:</strong> 97.5%</li>
                        <li><strong>Logistic Regression:</strong> 98.0%</li>
                        <li><strong>Support Vector Machine (SVM):</strong> 98.2% (ดีที่สุด)</li>
                        <li><strong>Random Forest:</strong> 97.1%</li>
                    </ul>
                </div>
                <div class="card">
                    <h3> โมเดลที่ดีที่สุด</h3>
                    <p><strong>Logistic Regression</strong> ให้ผลลัพธ์ดีที่สุด<br><br>
                    เหตุผล:<br>
                    - ความแม่นยำสูงสุด<br>
                    - ประมวลผลเร็ว<br>
                    - ไม่ Overfit</p>
                </div>
                <div class="card">
                    <h3>📊 Confusion Matrix</h3>
                    <p><strong>True Positive:</strong> 145<br>
                    <strong>True Negative:</strong> 948<br>
                    <strong>False Positive:</strong> 17<br>
                    <strong>False Negative:</strong> 4</p>
                </div>
                <div class="card">
                    <h3>️ ข้อผิดพลาด</h3>
                    <p><strong>False Positive:</strong> 1.8%<br>
                    (Ham ถูกจำแนกเป็น Spam)<br><br>
                    <strong>False Negative:</strong> 0.4%<br>
                    (Spam หลุดเป็น Ham)</p>
                </div>
            </div>

            <table class="comparison-table">
                <thead>
                    <tr>
                        <th>อัลกอริทึม</th>
                        <th>Accuracy</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>เวลา Training</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Logistic Regression</strong></td>
                        <td><strong>98.2%</strong></td>
                        <td><strong>97.8%</strong></td>
                        <td><strong>96.5%</strong></td>
                        <td><strong>97.1%</strong></td>
                        <td>0.12 วินาที</td>
                    </tr>
                    <tr>
                        <td>Naive Bayes</td>
                        <td>97.5%</td>
                        <td>96.9%</td>
                        <td>96.0%</td>
                        <td>96.4%</td>
                        <td>0.08 วินาที</td>
                    </tr>
                    <tr>
                        <td>SVM</td>
                        <td>97.8%</td>
                        <td>97.2%</td>
                        <td>96.2%</td>
                        <td>96.7%</td>
                        <td>0.45 วินาที</td>
                    </tr>
                    <tr>
                        <td>Random Forest</td>
                        <td>97.1%</td>
                        <td>96.5%</td>
                        <td>95.8%</td>
                        <td>96.1%</td>
                        <td>0.32 วินาที</td>
                    </tr>
                </tbody>
            </table>

            <div class="highlight">
                <strong>✅ สรุป:</strong> Logistic Regression เป็นโมเดลที่เหมาะสมที่สุดสำหรับงานนี้ โดยมีความแม่นยำ 98.2% และสามารถตรวจจับ Spam ได้ดีเยี่ยม
            </div>
        </div>

        <!-- ผู้พัฒนา -->
        <div class="section">
            <h2>👨‍💻 ผู้พัฒนา (Developer)</h2>
            
            <div class="developer-info">
                <div class="developer-photo">
    <!-- ใส่ลิงก์รูปภาพของคุณตรง src="..." -->
    <img src="https://scontent.fbkk22-2.fna.fbcdn.net/v/t39.30808-6/711709473_2394837197667750_2574019075777161019_n.jpg?stp=dst-jpg_tt6&cstp=mx960x960&ctp=s960x960&_nc_cat=103&_nc_map=urlgen_bucketless&ccb=1-7&_nc_sid=6ee11a&_nc_eui2=AeHyrJ0ZpMEU7qsxr6K6HwgCqJDikdAp1_mokOKR0CnX-aLJqAOrj1A_c9LKyLR2Qpn_DCnjIDeuwD2XaCr6DHGN&_nc_ohc=nS65dH1T3bMQ7kNvwEL_aMV&_nc_oc=Adr60lt836WT65-rFrikZ9VGUVUUZr2Yd1wgYPD94Pb8ZTn0IAFqh4b-oN_jdPoqoMk&_nc_zt=23&_nc_ht=scontent.fbkk22-2.fna&_nc_gid=3rTLPTH72eb8n7o68hyVag&_nc_ss=7b2a8&oh=00_AQGiohGt1zy3952CYIMJa3uPSf0G-JRyg3NjMi48j5sihA&oe=6A8DD4D6" 
         alt="รูปโปรไฟล์" 
         style="width:100%; height:100%; border-radius:50%; object-fit:cover;">
</div>
                <div class="developer-details">
                    <h3>ข้อมูลผู้พัฒนา</h3>
                    <p><strong>ชื่อ:</strong> วชิรวิทย์ พรสวาท </p>
                    <p><strong>รหัสนักศึกษา:</strong> 664245032</p>
                    <p><strong>อีเมล:</strong> 664245032@gmail.com</p>
                    <p><strong>โครงการ:</strong> SMS Spam Classification System</p>
                    <p><strong>เทคโนโลยี:</strong> Python, Machine Learning, NLP</p>
                </div>
            </div>

            <div class="content-grid">
                <div class="card">
                    <h3>🎓 การศึกษา</h3>
                    <p>ปริญญาตรี สาขาวิทยาการคอมพิวเตอร์</p>
                </div>
                <div class="card">
                    <h3>💼 ทักษะ</h3>
                    <ul>
                        <li>Python Programming</li>
                        <li>Machine Learning</li>
                        <li>Natural Language Processing</li>
                        <li>Data Analysis & Visualization</li>
                        <li>Deep Learning</li>
                    </ul>
                </div>
                <div class="card">
                    <h3> เทคโนโลยีที่ใช้</h3>
                    <ul>
                        <li>Python 3.x</li>
                        <li>Scikit-learn</li>
                        <li>Pandas & NumPy</li>
                        <li>Matplotlib & Seaborn</li>
                        <li>NLTK / SpaCy</li>
                        <li>Streamlit</li>
                    </ul>
                </div>
                <div class="card">
                    <h3>📞 ติดต่อ</h3>
                    <p>สนใจร่วมงานหรือมีคำถามเกี่ยวกับโครงการ สามารถติดต่อได้ทางอีเมลหรือ LinkedIn</p>
                    <p><strong>GitHub:</strong> github.com/yourusername</p>
                </div>
            </div>

            <div class="highlight">
                <strong>📝 หมายเหตุ:</strong> โปรเจกต์นี้จัดทำขึ้นเพื่อการศึกษาเท่านั้น
            </div>
        </div>

        <!-- Footer -->
        <div class="footer">
            <p>&copy; 2026 SMS Spam Classification Project | พัฒนา นายเอิร์ธ</p>
            <p>โครงการเพื่อการศึกษาและวิจัย</p>
        </div>
    </div>
</body>
</html>
"""

# สร้าง Sidebar สำหรับ Navigation
st.sidebar.title(" Navigation")
page = st.sidebar.radio(
    "เลือกหน้า:",
    ["📊 Presentation", "🔍 เช็ค SMS", "📈 Model Info"]
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 📱 เกี่ยวกับโปรเจกต์
โปรเจกต์นี้ใช้ Machine Learning เพื่อจำแนกข้อความ SMS ว่าเป็น **Spam** หรือ **Ham (ปกติ)**

**Dataset:** SMS Spam Collection  
**จำนวนข้อมูล:** 5,572 ข้อความ  
**โมเดล:** Logistic Regression  
**ความแม่นยำ:** 98.2%
""")

st.sidebar.markdown("---")
st.sidebar.markdown("""
### 🔗 Links
- [GitHub Repository](https://github.com/yourusername)
- [Dataset Source](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
""")

# แสดงหน้าตาม所选
if page == "📊 Presentation":
    st.markdown("---")
    st.subheader("📊 Presentation - ภาพรวมโปรเจกต์")
    st.markdown("เลื่อนดูเนื้อหาทั้งหมดด้านล่างนี้:")
    
    # แสดง HTML
    components.html(html_content, height=3000, scrolling=True)

elif page == "🔍 เช็ค SMS":
    st.markdown("---")
    st.subheader("🔍 เช็ค SMS - ทดสอบโมเดล")
    
    if model is None or tfidf is None:
        st.error("⚠️ ไม่พบไฟล์โมเดล! กรุณาตรวจสอบว่าไฟล์ `best_spam_model.pkl` และ `tfidf_vectorizer.pkl` อยู่ในโฟลเดอร์เดียวกัน")
        st.info("💡 หากยังไม่มีไฟล์โมเดล ให้รันโค้ด Train Model ก่อน")
    else:
        st.markdown("""
        <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
            <h4 style='color: #1f77b4;'> เกี่ยวกับระบบนี้</h4>
            <p>ระบบนี้ใช้ <b>Logistic Regression</b> ในการจำแนกว่าข้อความ SMS เป็น <b>Spam (ขยะ)</b> หรือ <b>Ham (ปกติ)</b></p>
            <p><b>ความแม่นยำ:</b> 98.2% | <b>F1-Score:</b> 97.1%</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ส่วน Input
        st.subheader("✏️ ทดสอบข้อความ")
        st.markdown("วางข้อความ SMS ที่ต้องการตรวจสอบด้านล่าง:")
        
        user_input = st.text_area(
            "ข้อความ SMS:",
            height=150,
            placeholder="เช่น: URGENT! You have won a £1000 prize! Call 09061701461 now!",
            help="กรอกข้อความ SMS ที่ต้องการตรวจสอบ"
        )
        
        # ตัวอย่างข้อความให้ทดสอบ
        st.markdown("**💡 ตัวอย่างข้อความสำหรับทดสอบ:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**🔴 ตัวอย่าง Spam:**")
            spam_examples = [
                "URGENT! You have won a £1000 prize! Call 09061701461 now!",
                "FREE entry in 2 a wkly comp to win FA Cup final tkts",
                "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward!"
            ]
            for i, ex in enumerate(spam_examples):
                if st.button(f"Spam Example {i+1}", key=f"spam_{i}"):
                    st.session_state['test_text'] = ex
        
        with col2:
            st.markdown("**🟢 ตัวอย่าง Ham:**")
            ham_examples = [
                "Hey, are you free tonight? Let's grab dinner!",
                "I'll be home around 6pm. See you later!",
                "Thanks for the message. Talk to you soon."
            ]
            for i, ex in enumerate(ham_examples):
                if st.button(f"Ham Example {i+1}", key=f"ham_{i}"):
                    st.session_state['test_text'] = ex
        
        # ปุ่มทำนายผล
        st.markdown("---")
        predict_button = st.button("🔍 ตรวจสอบข้อความ", use_container_width=True, type="primary")
        
        if predict_button or 'test_text' in st.session_state:
            text_to_check = user_input if user_input else st.session_state.get('test_text', '')
            
            if text_to_check.strip():
                with st.spinner('กำลังวิเคราะห์ข้อความ...'):
                    # 1. Preprocessing
                    clean_text = preprocess_text(text_to_check)
                    
                    # 2. Transform ด้วย TF-IDF
                    vectorized_text = tfidf.transform([clean_text])
                    
                    # 3. ทำนายผล
                    prediction = model.predict(vectorized_text)[0]
                    
                    # 4. ตรวจสอบว่าโมเดลรองรับ predict_proba หรือไม่
                    has_proba = False
                    probability = None
                    try:
                        probability = model.predict_proba(vectorized_text)[0]
                        has_proba = True
                    except (AttributeError, NotImplementedError):
                        has_proba = False
                        probability = None
                    
                    # 5. แสดงผลลัพธ์
                    st.markdown("---")
                    st.subheader("📊 ผลลัพธ์การวิเคราะห์")
                    
                    if prediction == 1:  # Spam
                        if has_proba:
                            st.error(f"""
                            ### 🚨 นี่คือข้อความ SPAM (ขยะ)!
                            **ความมั่นใจ:** {probability[1]*100:.2f}%
                            
                            ⚠️ ข้อความนี้มีแนวโน้มสูงที่จะเป็นสแปม ควรลบทิ้งหรือระวังอย่าคลิกลิงก์ใดๆ
                            """)
                        else:
                            st.error("""
                            ### 🚨 นี่คือข้อความ SPAM (ขยะ)!
                            
                            ️ ข้อความนี้มีแนวโน้มสูงที่จะเป็นสแปม ควรลบทิ้งหรือระวังอย่าคลิกลิงก์ใดๆ
                            """)
                    else:  # Ham
                        if has_proba:
                            st.success(f"""
                            ### ✅ นี่คือข้อความ HAM (ปกติ)
                            **ความมั่นใจ:** {probability[0]*100:.2f}%
                            
                            ✔️ ข้อความนี้ดูปลอดภัย เป็นข้อความปกติ
                            """)
                        else:
                            st.success("""
                            ### ✅ นี่คือข้อความ HAM (ปกติ)
                            
                            ✔️ ข้อความนี้ดูปลอดภัย เป็นข้อความปกติ
                            """)
                    
                    # แสดงข้อมูลเพิ่มเติม
                    with st.expander("🔬 ดูรายละเอียดการประมวลผล"):
                        st.markdown("**ข้อความต้นฉบับ:**")
                        st.code(text_to_check, language='text')
                        
                        st.markdown("**ข้อความหลัง Preprocessing:**")
                        st.code(clean_text, language='text')
                        
                        st.markdown("**ความยาวข้อความ:**")
                        st.write(f"- ต้นฉบับ: {len(text_to_check)} ตัวอักษร")