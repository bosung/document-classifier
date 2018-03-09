
def parse(text_list):
    document = {}
    for i, line in enumerate(text_list):
        if line.find('#DocID') >= 0:
            document['docid'] = line.split(":")[1].strip()
        elif line.find('#CAT\'03') >= 0:
            document['cat03'] = line.split(":")[1].strip()
        elif line.find('#CAT\'07') >= 0:
            document['cat07'] = line.split(":")[1].strip()
        elif line.find('#TITLE') >= 0:
            document['title'] = line.split(":")[1].strip()
        elif line.find('#TEXT') >= 0:
            document['text'] = [t for t in text_list[i+1:] if t != '' and t != '\n']

    return document


def text2dics(file_path):
    # create text stream
    f = open(file_path, "r", encoding='utf-8')

    text = f.read()
    text_lines = [t.strip() for t in text.split('\n')]

    documents = list()

    start_doc_idx = -1
    next_doc_idx = 0
    for i, line in enumerate(text_lines):
        if line == '@DOCUMENT' and start_doc_idx < 0:
            start_doc_idx = i
        elif line == '@DOCUMENT' and start_doc_idx >= 0:
            next_doc_idx = i
            text_doc = text_lines[start_doc_idx:next_doc_idx]
            documents.append(parse(text_doc))
            start_doc_idx = next_doc_idx

    # last document
    documents.append(parse(text_lines[next_doc_idx:]))

    f.close()
    return documents

