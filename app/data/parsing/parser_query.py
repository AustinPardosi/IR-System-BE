import json

with open(r"D:\D\Kuliah\Kuliah Semester 8\IF4042\IR-System-BE\parsing\query.text", "r", encoding="utf-8") as file:
    lines = file.readlines()

queries = {}
current_query = {}
current_field = None
current_id = None

for line in lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith(".I "):
        if current_id:
            queries[current_id] = current_query

        current_id = str(int(line[3:]))
        current_query = {
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
            current_query[current_field] += "" + line.strip()

if current_id:
    queries[current_id] = current_query

with open(r"D:\D\Kuliah\Kuliah Semester 8\IF4042\IR-System-BE\parsing\parsing_query.json", "w", encoding="utf-8") as out_file:
    json.dump(queries, out_file, indent=2, ensure_ascii=False)

print("Saved")