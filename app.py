from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import sqlglot

# ===============================
# 🔥 LOAD MODEL
# ===============================
model_path = "./fine_tune_model"

tokenizer = T5Tokenizer.from_pretrained(model_path)
model = T5ForConditionalGeneration.from_pretrained(model_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ===============================
# 🔥 VALIDATION FUNCTIONS
# ===============================
def validate_syntax(sql):
    try:
        sqlglot.parse_one(sql, dialect="tsql")
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def semantic_check(question, sql):
    q = question.lower()
    s = sql.lower()

    if any(k in q for k in [">", "<", "greater", "less", "between"]) and "where" not in s:
        return False, "Missing WHERE"

    if "count" in q and "count" not in s:
        return False, "Missing COUNT"

    if "top" in q and "top" not in s:
        return False, "Missing TOP"

    return True, "OK"

def validate_sql_pipeline(question, sql):
    ok, msg = validate_syntax(sql)
    if not ok:
        return False, msg

    ok, msg = semantic_check(question, sql)
    if not ok:
        return False, msg

    return True, "Valid SQL"

# ===============================
# 🔥 GENERATE SQL
# ===============================
def generate_sql(nl_query, schema):

    input_text = f"""
Translate English to Microsoft SQL Server query.

Query: {nl_query}

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

    sql = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # 🔥 VALIDATION
    is_valid, msg = validate_sql_pipeline(nl_query, sql)

    if is_valid:
        return sql
    else:
        print("⚠️ Validation failed:", msg)
        table = schema.split(":")[0]
        return f"SELECT * FROM {table};"

# ===============================
# 🔥 USER LOOP
# ===============================
while True:
    print("\n--- TEXT TO SQL GENERATOR (SQL Server) ---")

    nl_query = input("Enter your query: ")
    schema = input("Enter schema (e.g., Students: id, name, marks): ")

    sql = generate_sql(nl_query, schema)

    print("\n✅ Generated SQL:")
    print(sql)

    cont = input("\nContinue? (y/n): ")
    if cont.lower() != 'y':
        break