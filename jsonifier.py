import re
import json

def is_heading_line(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    # Bold-only heading like **TITLE**
    if re.match(r'^\*\*(.+?)\*\*$', s):
        return True
    # Markdown heading
    if s.startswith('#'):
        return True
    # ALL-CAPS heuristic (no lowercase letters, not a sentence)
    stripped = re.sub(r'[^A-Za-z0-9\s/&()\-.,]', '', s)
    if stripped and (re.sub(r'[^A-Z0-9\s/&()\-]', '', s) == stripped) and len(s.split()) <= 12 and not s.endswith('.'):
        # contains at least one letter and is essentially uppercase
        return True
    return False

def extract_heading_text(s: str) -> str:
    s = s.strip()
    m = re.match(r'^\*\*(.+?)\*\*$', s)
    if m:
        return m.group(1).strip()
    if s.startswith('#'):
        return s.lstrip('#').strip()
    return s.strip()

def normalize_para(p: str) -> str:
    # collapse internal multiple spaces, but keep line-break bullets
    p = p.strip()
    # convert CRLF and mixed newlines -> \n
    p = p.replace('\r\n', '\n').replace('\r', '\n')
    # replace repeated whitespace with single space except preserve '- ' bullets newlines
    lines = [ln.strip() for ln in p.splitlines()]
    if len(lines) == 1:
        return re.sub(r'\s+', ' ', lines[0]).strip()
    # preserve bullet lines separated by newline
    return '\n'.join(re.sub(r'\s+', ' ', ln).strip() for ln in lines if ln.strip())

def parse_document(text: str):
    paragraphs = re.split(r'\n\s*\n', text.strip())
    sections = []

    current_heading = None
    current_content = None  # dict holding '_text' and bold keys
    last_bold_key = None

    bold_re = re.compile(r'\*\*(?P<term>.+?)\*\*')

    def start_section(heading):
        nonlocal current_heading, current_content, last_bold_key
        if current_heading is not None:
            # finalize previous section
            # if only _text exists (no bold keys), convert to list for backward compatibility
            keys = [k for k in current_content.keys() if k != "_text"]
            if keys:
                # keep dict; remove _text if empty
                if current_content.get("_text"):
                    current_content["_text"] = [t for t in current_content["_text"]]
                else:
                    current_content.pop("_text", None)
            else:
                # convert to list
                current_content = current_content.get("_text", [])
            sections.append({"heading": current_heading, "content": current_content})
        current_heading = heading
        current_content = {"_text": []}
        last_bold_key = None
        return

    for para in paragraphs:
        s = para.strip()
        if not s:
            continue

        if is_heading_line(s):
            heading = extract_heading_text(s)
            start_section(heading)
            continue

        # If no heading yet, create a default root heading (document title or ROOT)
        if current_heading is None:
            start_section("ROOT")

        para_norm = normalize_para(s)

        # Find bold terms in paragraph
        bold_matches = list(bold_re.finditer(s))
        if bold_matches:
            # If paragraph contains one or more bolded terms, split their definitions
            for idx, m in enumerate(bold_matches):
                term = m.group("term").strip()
                # determine text after this match until next bold or end
                start_idx = m.end()
                end_idx = bold_matches[idx+1].start() if idx+1 < len(bold_matches) else len(s)
                def_text_raw = s[start_idx:end_idx].strip()
                # clean common leading words like 'means', ':', '-'
                def_text = re.sub(r'^[\s:–—\-]*', '', def_text_raw)
                def_text = re.sub(r'^(means|mean|is|are)\b[:\s\-]*', '', def_text, flags=re.IGNORECASE)
                def_text = normalize_para(def_text) if def_text else ""

                # if the definition is empty but paragraph has more text beyond bold, fallback to paragraph normalized
                if not def_text:
                    # grab the rest of the paragraph if any
                    def_text = para_norm
                # store
                current_content.setdefault(term, "")
                # if there was already content for this term, append with newline
                if current_content[term]:
                    current_content[term] = current_content[term].rstrip() + "\n" + def_text
                else:
                    current_content[term] = def_text
                last_bold_key = term
            continue  # processed this paragraph

        # No bold in this paragraph
        # If this paragraph looks like a bullet list or numbered list, attach to last_bold_key (if any),
        # otherwise if last_bold_key exists - treat as continuation; else store in _text.
        is_bullet = bool(re.match(r'^\s*[-•\u2022]\s+', s) or re.match(r'^\s*\d+\.\s+', s))
        if last_bold_key and (is_bullet or True):
            # attach as continuation to last bold key
            cont = para_norm
            # if it begins with a bullet marker, try to strip the leading bullet for nicer storage but preserve newlines
            # preserve bullets inside string
            if is_bullet:
                # keep as-is but normalized
                pass
            # append to existing definition
            current_content.setdefault(last_bold_key, "")
            if current_content[last_bold_key]:
                current_content[last_bold_key] = current_content[last_bold_key].rstrip() + "\n\n" + cont
            else:
                current_content[last_bold_key] = cont
        else:
            # store as ordinary paragraph text under _text
            current_content.setdefault("_text", [])
            current_content["_text"].append(para_norm)

    # finalize last section
    if current_heading is not None:
        keys = [k for k in current_content.keys() if k != "_text"]
        if keys:
            if current_content.get("_text"):
                current_content["_text"] = [t for t in current_content["_text"]]
            else:
                current_content.pop("_text", None)
        else:
            current_content = current_content.get("_text", [])
        sections.append({"heading": current_heading, "content": current_content})

    return sections

if __name__ == "__main__":
    # Usage: replace 'document.txt' with your file
    with open(r"C:\Users\prady\OneDrive\Desktop\PeakRX\data.txt", "r", encoding="cp1252") as f:
        raw = f.read()

    parsed = parse_document(raw)
    # pretty print / write file
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    with open("ICICI.json", "w", encoding="cp1252") as out:
        json.dump(parsed, out, indent=2, ensure_ascii=False)
