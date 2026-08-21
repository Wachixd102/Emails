import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ดาวน์โหลด NLTK data (รันครั้งแรกครั้งเดียว)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# โหลดโมเดล SVM ที่ดีที่สุด และ TF-IDF Vectorizer ที่เราบันทึกไว้
@st.cache_resource
def load_models():
    model = joblib.load('Support_Vector_Machine_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_models()

# เตรียมเครื่องมือสำหรับ Preprocessing
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# ฟังก์ชันทำความสะอาดข้อความ (ต้องเหมือนกับตอน Train เป๊ะๆ)
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ==========================================
# ส่วนของ UI หน้าเว็บ Streamlit
# ==========================================
st.set_page_config(page_title="Spam Email Detector", page_icon="📧", layout="centered")

st.title("📧 Email Spam Classifier")
st.write("ระบบตรวจจับอีเมลขยะ (Spam) ด้วย Machine Learning (SVM)")
st.markdown("---")

# ช่องให้ผู้ใช้กรอกข้อความ
user_input = st.text_area(
    "วางข้อความอีเมลของคุณด้านล่าง:", 
    height=150, 
    placeholder="เช่น: Congratulations! You have won a $1000 Walmart gift card. Click here to claim..."
)

# ปุ่มทำนายผล
if st.button("🔍 ตรวจสอบอีเมล", use_container_width=True):
    if user_input.strip():
        # 1. Preprocessing ข้อความที่ผู้ใช้กรอก
        clean_text = preprocess_text(user_input)
        
        # 2. แปลงข้อความเป็นตัวเลขด้วย TF-IDF
        vectorized_text = tfidf.transform([clean_text])
        
        # 3. ทำนายผล
        prediction = model.predict(vectorized_text)[0]
        probability = model.predict_proba(vectorized_text)[0]
        
        # 4. แสดงผลลัพธ์
        st.markdown("---")
        st.subheader("📊 ผลลัพธ์การวิเคราะห์:")
        
        if prediction == 1:
            st.error(f"🚨 **นี่คืออีเมลขยะ (SPAM)!**")
            st.warning(f"⚠️ ความมั่นใจของโมเดล: **{probability[1]*100:.2f}%**")
        else:
            st.success(f"✅ **นี่คืออีเมลปกติ (HAM)**")
            st.info(f"💡 ความมั่นใจของโมเดล: **{probability[0]*100:.2f}%**")
            
        # แสดงข้อความที่ผ่านการ Clean แล้ว (โชว์ว่าเราทำ Preprocessing จริงๆ)
        with st.expander("🔍 ดูข้อความหลังการ Preprocessing"):
            st.code(clean_text)
    else:
        st.warning("⚠️ กรุณากรอกข้อความก่อนทำการตรวจสอบ")

# Footer
st.markdown("---")
st.caption("Developed by [ชื่อ-นามสกุล ของคุณ] | Mini Project: Email Spam Classification")