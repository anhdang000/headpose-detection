from difflib import SequenceMatcher

def calc_score(a, b):
    return SequenceMatcher(None, ''.join(a), ''.join(b)).ratio()
    
def filter_records(record, patience=2):
    buff = []
    result = []
    for element in record:
        if element != 'F':
            buff = buff[1:] + [element]
            if len(set(buff)) == 1 and len(buff) == patience:
                if len(result) >= 1:
                    if element != result[-1]:
                        result.append(element)
                else:
                    result.append(element)
            

            elif len(buff) < patience:
                buff.append(element)
    return result

if __name__ == "__main__":
    record = ['F', 'F', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'U', 'D', 'D']
    print(filter_records(record))