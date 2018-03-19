from model import Document

def parse(text_list):
    for i, line in enumerate(text_list):
        if line.find('#DocID') >= 0:
            docid = line.split(":")[1].strip()
        elif line.find('#CAT\'03') >= 0:
            category = line.split(":")[1].strip()
        elif line.find('#TITLE') >= 0:
            title = line.split(":")[1].strip()
        elif line.find('#TEXT') >= 0:
            text = [t for t in text_list[i+1:] if t != '' and t != '\n']

    return Document(docid, category, title, text)


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


def get_category_dict(file_path):
    """read category list file and return 'category name' to 'id' dictionary
    """
    f = open(file_path, "r", encoding='utf-8')

    category_dict = {}
    category_id = 0
    for line in f.readlines():
        if line[0] == '#':
            continue
        cat = {}
        cat['doc_num'], cat['root'], cat['middle'], cat['leaf'] = \
                [x.strip() for x in line.split("/")]
        cat['cat_id'] = category_id
        category_id += 1
        category_dict[cat['leaf']] = cat

    f.close()
    return category_dict

