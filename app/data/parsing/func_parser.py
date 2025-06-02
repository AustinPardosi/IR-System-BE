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