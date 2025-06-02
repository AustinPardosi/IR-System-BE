def parser_qrels(filename):
    import json
    with open(filename, "r", encoding="utf-8") as file:
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

    return(qrels_dict)

def parser_docs(filename):
    import json

    with open(filename, "r", encoding="utf-8") as file:
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

        elif line == ".T" or line == ".A" or line == ".W" or line == ".B":
            pass

        elif line == ".X":
            neglect = True

        else:
            if (not neglect):
                current_doc += " " + line.strip()

    if current_id:
        docs[current_id] = current_doc

    return(docs)

def parser_query(filename):
    import json

    with open(filename, "r", encoding="utf-8") as file:
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
    
    return(queries)