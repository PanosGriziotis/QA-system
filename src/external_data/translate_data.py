import deepl

auth_key = ""  # Replace with your key
translator = deepl.Translator(auth_key)

t_lines = []
with open ("who_en.txt", "r") as fp:
    lines= fp.readlines()
    
    for line in lines:
        c_line = line.strip()
        result = translator.translate_text(c_line, target_lang="EL")         
        result = str(result)
        t_lines.append(result)

with open ("who_el.txt", "a") as fp:
    for line in t_lines:
        fp.write(line + "\n")