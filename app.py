import streamlit as st
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ดาวน์โหลด NLTK data (รันครั้งแรกครั้งเดียว)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt', quiet=True)

# ==========================================
# โหลดโมเดลและ Vectorizer
# ==========================================
@st.cache_resource
def load_models():
    model = joblib.load('best_spam_model.pkl')
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf

model, tfidf = load_models()

# ==========================================
# ฟังก์ชัน Preprocessing (ต้องเหมือนตอน Train เป๊ะ!)
# ==========================================
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

# ==========================================
# UI ของ Streamlit
# ==========================================
st.set_page_config(
    page_title="📧 SMS Spam Classifier",
    page_icon="📧",
    layout="centered"
)

# Header
st.title("📧 SMS Spam Classifier")
st.markdown("---")
st.markdown("""
<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px;'>
    <h4 style='color: #1f77b4;'>🤖 เกี่ยวกับโปรเจกต์นี้</h4>
    <p>ระบบนี้ใช้ Machine Learning (Support Vector Machine - SVM) 
    ในการจำแนกว่าข้อความ SMS เป็น <b>Spam (ขยะ)</b> หรือ <b>Ham (ปกติ)</b></p>
    <p><b>Dataset:</b> SMS Spam Collection Dataset (5,572 messages)</p>
    <p><b>Preprocessing:</b> Lowercase → Remove Punctuation → Remove Stopwords → Lemmatization → TF-IDF</p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar - แสดงข้อมูลโมเดล
with st.sidebar:
    st.header("📊 ข้อมูลโมเดล")
    st.markdown("**Algorithm:** Support Vector Machine (SVM)")
    st.markdown("**Features:** TF-IDF (3,000 features)")
    st.markdown("**Dataset:** 5,572 SMS messages")
    st.markdown("- Ham (ปกติ): 4,825")
    st.markdown("- Spam (ขยะ): 747")
    st.markdown("---")
    st.markdown("**ผู้พัฒนา:** [ใส่ชื่อ-นามสกุล ของคุณ]")
    st.markdown("**รหัสนักศึกษา:** [ใส่รหัส]")

# ==========================================
# ส่วน Input
# ==========================================
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

# ==========================================
# ปุ่มทำนายผล
# ==========================================
st.markdown("---")
predict_button = st.button("🔍 ตรวจสอบข้อความ", use_container_width=True, type="primary")

if predict_button:
    if user_input.strip():
        with st.spinner('กำลังวิเคราะห์ข้อความ...'):
            # 1. Preprocessing
            clean_text = preprocess_text(user_input)
            
            # 2. Transform ด้วย TF-IDF
            vectorized_text = tfidf.transform([clean_text])
            
            # 3. ทำนายผล
            prediction = model.predict(vectorized_text)[0]
            probability = model.predict_proba(vectorized_text)[0]
            
            # 4. แสดงผลลัพธ์
            st.markdown("---")
            st.subheader("📊 ผลลัพธ์การวิเคราะห์")
            
            if prediction == 1:  # Spam
                st.error(f"""
                ### 🚨 นี่คือข้อความ SPAM (ขยะ)!
                **ความมั่นใจ:** {probability[1]*100:.2f}%
                
                ️ ข้อความนี้มีแนวโน้มสูงที่จะเป็นสแปม ควรลบทิ้งหรือระวังอย่าคลิกลิงก์ใดๆ
                """)
            else:  # Ham
                st.success(f"""
                ### ✅ นี่คือข้อความ HAM (ปกติ)
                **ความมั่นใจ:** {probability[0]*100:.2f}%
                
                ✔️ ข้อความนี้ดูปลอดภัย เป็นข้อความปกติ
                """)
            
            # แสดงข้อมูลเพิ่มเติม
            with st.expander("🔬 ดูรายละเอียดการประมวลผล"):
                st.markdown("**ข้อความต้นฉบับ:**")
                st.code(user_input, language='text')
                
                st.markdown("**ข้อความหลัง Preprocessing:**")
                st.code(clean_text, language='text')
                
                st.markdown("**ความยาวข้อความ:**")
                st.write(f"- ต้นฉบับ: {len(user_input)} ตัวอักษร")
                st.write(f"- หลัง cleaning: {len(clean_text)} ตัวอักษร")
                
                st.markdown("**ความน่าจะเป็น:**")
                st.write(f"- Spam: {probability[1]*100:.2f}%")
                st.write(f"- Ham: {probability[0]*100:.2f}%")
    else:
        st.warning("️ กรุณากรอกข้อความก่อนทำการตรวจสอบ")

# ==========================================
# Footer
# ==========================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>📧 SMS Spam Classification Project | Mini Project 2026</p>
    <p>Developed with ❤️ using Python, Scikit-Learn, and Streamlit</p>
</div>
""", unsafe_allow_html=True)