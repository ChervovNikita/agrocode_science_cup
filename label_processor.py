import pymorphy2
morph = pymorphy2.MorphAnalyzer()


def delete_double_spaces(nm):
    new_nm = ''
    for char in nm:
        if char != ' ' or (len(new_nm) and new_nm[-1] != ' '):
            new_nm += char
    return new_nm.strip()


def label_process(nm):
    nm = nm.lower()

    new_nm = ''
    balance = 0
    for char in nm:
        if char in '({[':
            balance += 1
        elif char in ')}]':
            balance = max(0, balance - 1)
        elif balance == 0:
            new_nm += char
    nm = new_nm

    new_nm = ''
    for char in nm:
        if char in 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя ':
            new_nm += char
    nm = delete_double_spaces(new_nm)

    new_nm = ''
    black_list = ['х', 'хх', 'ххх', 'мм', 'с', 'км', 'от', 'в', 'т']
    for word in nm.split(' '):
        if word not in black_list and len(word) > 2:
            state = morph.parse(word)[0]
            tag = state.tag
            word = state.normal_form
            if 'NOUN' in tag or 'ADJF' in tag:
                new_nm += word + ' '
    nm = delete_double_spaces(new_nm)

    return nm
