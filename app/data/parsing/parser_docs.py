import json

with open(r"app\data\parsing\cisi.all", "r", encoding="utf-8") as file:
    lines = file.readlines()

docs = {}
current_doc = {}
current_field = None
current_id = None

for line in lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith(".I "):
        if current_id:
            docs[current_id] = current_doc

        current_id = str(int(line[3:]))
        current_doc = {
            "title": "",
            "author": "",
            "words": "",
            "bibliographic": ""
        }
        current_field = None

    elif line == ".T":
        current_field = "title"
    
    elif line == ".A":
        current_field = "author"

    elif line == ".W":
        current_field = "words"

    elif line == ".B":
        current_field = "bibliographic"

    else:
        if current_field:
            current_doc[current_field] += "" + line.strip()

if current_id:
    docs[current_id] = current_doc

with open(r"app\data\parsing\parsing_docs_2.json", "w", encoding="utf-8") as out_file:
    json.dump(docs, out_file, indent=2, ensure_ascii=False)

print("Saved")