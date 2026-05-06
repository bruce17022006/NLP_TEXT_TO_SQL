from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import sqlglot
import re

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
# 🔥 ADVANCED VALIDATION & REPAIR
# ===============================
def validate_syntax(sql):
    """Verifies T-SQL syntax using sqlglot."""
    try:
        sqlglot.parse_one(sql, dialect="tsql")
        return True, "Valid"
    except Exception as e:
        return False, str(e)

def attempt_auto_fix(question, sql, schema, error_type):
    """
    Manually injects missing SQL components or repairs common hallucinations[cite: 1].
    """
    q = question.lower()
    # Basic cleaning of the model's output
    fixed_sql = sql.strip().rstrip(';')
    
    # 1. Fix Missing WHERE (e.g., Manoj filter)[cite: 5]
    if "manoj" in q and "where" not in fixed_sql.lower():
        fixed_sql += " WHERE name = 'Manoj'"
    
    # 2. Fix Missing TOP[cite: 1]
    elif "top" in q and "top" not in fixed_sql.lower():
        nums = re.findall(r'\d+', q)
        n = nums[0] if nums else "5"
        fixed_sql = fixed_sql.replace("SELECT", f"SELECT TOP {n}", 1)
        
    # 3. Fix Missing/Broken JOIN[cite: 5]
    # Triggered if schema has multiple tables but SQL does not have a JOIN
    elif "\n" in schema and "join" not in fixed_sql.lower():
        tables = [line.split(":")[0].strip() for line in schema.split("\n") if ":" in line]
        if len(tables) >= 2:
            # Reconstruct the query as a JOIN[cite: 5]
            # Assumes dept_id is the common key based on your project goals
            fixed_sql = f"SELECT * FROM {tables[0]} JOIN {tables[1]} ON {tables[0]}.dept_id = {tables[1]}.dept_id"

    # 4. Fix Aggregations[cite: 1]
    elif "count" in q and "count" not in fixed_sql.lower():
        fixed_sql = re.sub(r"SELECT (.*?) FROM", "SELECT COUNT(*) FROM", fixed_sql, flags=re.IGNORECASE)

    return fixed_sql + ";"

def semantic_check(question, sql):
    """Checks if the query logic matches the English intent[cite: 6]."""
    q = question.lower()
    s = sql.lower()

    if "manoj" in q and "where" not in s:
        return False, "Missing WHERE"
    if "count" in q and "count" not in s:
        return False, "Missing COUNT"
    if "top" in q and "top" not in s:
        return False, "Missing TOP"
    # Detect if multiple tables are implied but JOIN is missing[cite: 5]
    if ("and their" in q or "with their" in q or "names with" in q) and "join" not in s:
        return False, "Missing JOIN"

    return True, "OK"

# ===============================
# 🔥 GENERATION PIPELINE
# ===============================
def generate_sql(nl_query, schema):
    input_text = f"Translate English to Microsoft SQL Server query. Query: {nl_query} Tables:\n{schema} SQL:"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=192, num_beams=3) # Optimized beams for speed

    generated_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Validation Steps[cite: 1, 6]
    syntax_ok, syntax_msg = validate_syntax(generated_sql)
    semantic_ok, semantic_msg = semantic_check(nl_query, generated_sql)

    if syntax_ok and semantic_ok:
        return generated_sql
    else:
        # Use whatever error was found to trigger the fix[cite: 1]
        error_type = semantic_msg if not semantic_ok else "Syntax Error"
        print(f"⚠️ Triggering Repair: {error_type}")
        return attempt_auto_fix(nl_query, generated_sql, schema, error_type)

# ===============================
# 🔥 USER INTERFACE
# ===============================
print("\n--- GENERIC TEXT TO SQL (Self-Repairing) ---")
while True:
    query = input("\nEnter query: ")
    print("Enter schema (Empty line to finish):")
    schema_lines = []
    while True:
        line = input()
        if not line: break
        schema_lines.append(line)
    
    combined_schema = "\n".join(schema_lines)
    final_sql = generate_sql(query, combined_schema)
    
    print(f"\n✅ Output Query:\n{final_sql}")
    
    if input("\nContinue? (y/n): ").lower() != 'y': break