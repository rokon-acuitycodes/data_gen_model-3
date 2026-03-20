import re

def get_paragraphs(text):
    """Simple paragraph splitter using regex."""
    paras = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paras if p.strip()]
