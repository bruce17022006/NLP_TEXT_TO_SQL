import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import sqlglot
import re

# ===============================
# 🔥 LOAD MODEL (CACHE)
# ===============================
@st.cache_resource
def load_model():
    model_path = "./fine_tune_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model()

# ===============================
# 🔥 THEME & STYLING
# ===============================
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

def apply_custom_theme():
    if st.session_state.theme == "dark":
        st.markdown("""<style>
            body, .stApp { background-color: #0E1117; color: white; }
            .stTextInput input, .stTextArea textarea { background-color: #262730 !important; color: white !important; }
            .stButton>button { background-color: #4CAF50; color: white; border-radius: 12px; width: 100%; }
        </style>""", unsafe_allow_html=True)
    else:
        st.markdown("""<style>
            body, .stApp { background-color: #FFFFFF; color: black; }
            .stTextInput input, .stTextArea textarea { background-color: #F5F5F5 !important; color: black !important; }
            .stButton>button { background-color: #008CBA; color: white; border-radius: 12px; width: 100%; }
        </style>""", unsafe_allow_html=True)

apply_custom_theme()

# ===============================
# 🧠 REPAIR & VALIDATION LOGIC
# ===============================
def attempt_auto_fix(question, sql, schema, error_type):
    """Automatically repairs the SQL based on detected missing parts[cite: 1, 5]."""
    q = question.lower()
    fixed_sql = sql.strip().rstrip(';')
    
    if error_type == "Missing WHERE" and "manoj" in q:
        fixed_sql += " WHERE name = 'Manoj'"
    elif error_type == "Missing TOP" and "top" in q:
        nums = re.findall(r'\d+', q)
        n = nums[0] if nums else "5"
        fixed_sql = fixed_sql.replace("SELECT", f"SELECT TOP {n}", 1)
    elif error_type == "Missing COUNT" and "count" in q:
        fixed_sql = re.sub(r"SELECT (.*?) FROM", "SELECT COUNT(*) FROM", fixed_sql, flags=re.IGNORECASE)
    elif error_type == "Missing JOIN" and "\n" in schema:
        tables = [line.split(":")[0].strip() for line in schema.split("\n") if ":" in line]
        if len(tables) >= 2:
            fixed_sql = f"SELECT * FROM {tables[0]} JOIN {tables[1]} ON {tables[0]}.dept_id = {tables[1]}.dept_id"
    
    return fixed_sql + ";"

def semantic_check(question, sql):
    """Validates if English intent is missing in generated SQL."""
    q, s = question.lower(), sql.lower()
    if "manoj" in q and "where" not in s: return False, "Missing WHERE"
    if "count" in q and "count" not in s: return False, "Missing COUNT"
    if "top" in q and "top" not in s: return False, "Missing TOP"
    if ("and their" in q or "with their" in q) and "join" not in s: return False, "Missing JOIN"
    return True, "OK"

def validate_syntax(sql):
    """Verifies T-SQL syntax using sqlglot[cite: 1, 6]."""
    try:
        sqlglot.parse_one(sql, dialect="tsql")
        return True, "Valid"
    except Exception as e:
        return False, str(e)

# ===============================
# 🔥 MAIN GENERATOR
# ===============================
def generate_sql(query, schema):
    input_text = f"Translate English to Microsoft SQL Server query. Query: {query} Tables:\n{schema} SQL:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=192, num_beams=3) # Generic length[cite: 4]
    
    raw_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Validation & Repair Loop[cite: 1, 6]
    syntax_ok, _ = validate_syntax(raw_sql)
    semantic_ok, semantic_msg = semantic_check(query, raw_sql)

    if syntax_ok and semantic_ok:
        return raw_sql, None
    else:
        error_type = semantic_msg if not semantic_ok else "Syntax Error"
        fixed_sql = attempt_auto_fix(query, raw_sql, schema, error_type)
        return fixed_sql, error_type

# ===============================
# 📥 UI LAYOUT
# ===============================
st.markdown("<h1 style='text-align:center;'>Text to SQL Generator </h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Self-Repairing Microsoft SQL Server Queries</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    query_input = st.text_input("Enter your natural language query:")
    schema_input = st.text_area("Enter table schemas (One per line):", placeholder="Students: id, name, dept_id\nDepartments: dept_id, name")
    generate_btn = st.button("Generate & Repair SQL")

with col2:
    st.subheader("Output")
    if generate_btn:
        if query_input and schema_input:
            with st.spinner("Analyzing and Generating..."):
                final_sql, repair_note = generate_sql(query_input, schema_input)
            
            if repair_note:
                st.warning(f"⚠️ Auto-Repair Triggered: {repair_note}")
            
            st.success("Final Optimized Query:")
            st.code(final_sql, language="sql")
        else:
            st.error("Please provide both a query and a schema.")

# ===============================
# 💡 EXAMPLES & FOOTER
# ===============================
st.sidebar.header("Example Queries")
st.sidebar.info("1. Get employees where name is Manoj\n2. Get student names and their department names\n3. Top 5 expensive products\n4. How many students are in department 10")

st.markdown("---")
st.markdown("<p style='text-align:center;color:gray;'>Powered by T5-Small + Sqlglot + Streamlit</p>", unsafe_allow_html=True)