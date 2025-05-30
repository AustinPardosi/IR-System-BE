import json

# Open and read the file
with open(r"D:\D\Kuliah\Kuliah Semester 8\IF4042\IR-System-BE\parsing\qrels.text", "r", encoding="utf-8") as file:
    lines = file.readlines()

qrels_dict = {}

for line in lines:
    line = line.strip()
    if not line:
        continue

    parts = line.split()
    if len(parts) != 4:
        continue

    query_id, doc_id, _, _ = parts

    if query_id not in qrels_dict:
        qrels_dict[query_id] = []

    qrels_dict[query_id].append(doc_id)

with open(r"D:\D\Kuliah\Kuliah Semester 8\IF4042\IR-System-BE\parsing\parsing_qrels.json", "w", encoding="utf-8") as out_file:
    json.dump(qrels_dict, out_file, indent=2, ensure_ascii=False)

print("Saved")