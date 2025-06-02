import json

with open(r"D:\Kuliah\Semester 8\IF4042 Sistem Temu Balik Informasi\Tubes\IR-System-BE\app\data\parsing\cisi.all", "r", encoding="utf-8") as file:
    lines = file.readlines()

docs = {}
current_doc = ""
current_id = None
neglect = False

for line in lines:
    line = line.strip()
    if not line:
        continue

    if line.startswith(".I "):
        neglect = False
        if current_id:
            docs[current_id] = current_doc

        current_id = str(int(line[3:]))
        current_doc = ""
        # current_field = None

    elif line == ".T" or line == ".W":
        neglect = False

    elif line == ".X" or line == ".A" or line == ".B":
        neglect = True

    else:
        if (not neglect):
            current_doc += " " + line.strip()

if current_id:
    docs[current_id] = current_doc

with open(r"D:\Kuliah\Semester 8\IF4042 Sistem Temu Balik Informasi\Tubes\IR-System-BE\app\data\parsing\parsing_docs.json", "w", encoding="utf-8") as out_file:
    json.dump(docs, out_file, indent=2, ensure_ascii=False)

print("Saved")