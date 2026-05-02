import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import re

# ===============================
# 🔥 LOAD MODEL (CACHE)
# ===============================
@st.cache_resource
def load_model():
    model_path = "./fine_tuned_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return tokenizer, model, device

tokenizer, model, device = load_model()

# ===============================
# 🔥 THEME STATE
# ===============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# ===============================
# 🔥 HEADER + TOGGLE
# ===============================
col1, col2 = st.columns([9,1])

with col2:
    if st.button("🌗", help="Toggle Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"

# ===============================
# 🎨 APPLY FULL THEME
# ===============================
if st.session_state.theme == "dark":
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #0E1117;
        color: white;
    }

    h1, h2, h3 {
        color: white;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #262730 !important;
        color: white !important;
    }

    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 12px;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }

    .stCodeBlock {
        background-color: #1E1E1E !important;
    }

    hr {
        border-color: #444;
    }
    </style>
    """, unsafe_allow_html=True)

else:
    st.markdown("""
    <style>
    body, .stApp {
        background-color: #FFFFFF;
        color: black;
    }

    h1, h2, h3 {
        color: black;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: #F5F5F5 !important;
        color: black !important;
    }

    .stButton>button {
        background-color: #008CBA;
        color: white;
        border-radius: 12px;
        height: 45px;
        width: 100%;
        font-size: 16px;
    }

    .stCodeBlock {
        background-color: #F0F0F0 !important;
    }

    hr {
        border-color: #CCC;
    }
    </style>
    """, unsafe_allow_html=True)

# ===============================
# 🧠 HEADER TEXT
# ===============================
st.markdown("<h1 style='text-align:center;'>Text to SQL Generator</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Convert Natural Language → Microsoft SQL Server Query</p>", unsafe_allow_html=True)

st.caption(f"Theme: {st.session_state.theme.upper()}")

st.markdown("---")

# ===============================
# 🔥 RULE ENGINE
# ===============================
def rule_based_sql(query, schema):
    q = query.lower()
    table = schema.split(":")[0].strip()

    if "count" in q or "how many" in q:
        return f"SELECT COUNT(*) FROM {table};"

    if "top" in q:
        nums = re.findall(r'\d+', q)
        n = nums[0] if nums else "5"
        return f"SELECT TOP {n} * FROM {table};"

    if "name" in q:
        return f"SELECT name FROM {table};"

    return None

# ===============================
# 🔥 MODEL GENERATION
# ===============================
def model_sql(query, schema):
    input_text = f"""
Translate English to Microsoft SQL Server query.

Query: {query}

Tables:
{schema}

SQL:
"""

    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=64,
            num_beams=5,
            do_sample=False
        )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.strip()

# ===============================
# 🔥 VALIDATION
# ===============================
def validate_sql(sql):
    if not sql.upper().startswith("SELECT"):
        return None
    if "FROM" not in sql.upper():
        return None
    return sql

# ===============================
# 🔥 MAIN GENERATOR
# ===============================
def generate_sql(query, schema):

    sql = rule_based_sql(query, schema)
    if sql:
        return sql

    sql = model_sql(query, schema)

    valid = validate_sql(sql)
    if valid:
        return valid

    table = schema.split(":")[0].strip()
    return f"SELECT * FROM {table};"

# ===============================
# 📥 INPUT / OUTPUT
# ===============================
col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    query = st.text_input("Enter your query:")
    schema = st.text_area("Enter schema (e.g., Student: id, name, age):")

    generate_btn = st.button("Generate SQL")

with col2:
    st.subheader("Output")

    if generate_btn:
        if query and schema:
            with st.spinner("Generating SQL..."):
                sql = generate_sql(query, schema)

            st.success("Generated SQL Query:")
            st.code(sql, language="sql")
        else:
            st.warning("Please enter both query and schema")

# ===============================
# 💡 EXAMPLES
# ===============================
st.markdown("---")
st.subheader("Example Queries")

examples = [
    "count students",
    "get student names",
    "top 3 students",
    "get students with marks > 80"
]

for ex in examples:
    st.write("👉", ex)

# ===============================
# 🔚 FOOTER
# ===============================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:gray;'>Built using NLP + Transformers + Streamlit</p>",
    unsafe_allow_html=True
)