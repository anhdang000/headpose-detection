from difflib import SequenceMatcher
import numpy as np

def calc_score(a, b):
    return SequenceMatcher(None, ''.join(a), ''.join(b)).ratio()
    
def filter_records(record, patience=2):
    buff = []
    result = []
    for element in record:
        buff = buff[1:] + [element]
        if len(set(buff)) == 1 and len(buff) == patience:
            if len(result) >= 1:
                if element != result[-1]:
                    result.append(element)
            else:
                result.append(element)
        

        elif len(buff) < patience:
            buff.append(element)
    result = [x for x in result if x != 'F']
    return result

def calc_distance(p1, p2):
    return np.sqrt((p2 - p1)[0]**2 + (p2 - p1)[1]**2)

def compute_EAR(points):
    # Left eye
    A1 = calc_distance(points[37], points[41])
    A2 = calc_distance(points[38], points[40])
    B = calc_distance(points[36], points[39])
    left_EAR = (A1 + A2) / (2*B)

    # Light eye
    A1 = calc_distance(points[43], points[47])
    A2 = calc_distance(points[44], points[46])
    B = calc_distance(points[42], points[45])
    right_EAR = (A1 + A2) / (2*B)

    return (left_EAR, right_EAR) / 2


if __name__ == "__main__":
    record = ['F', 'F', 'R', 'R', 'R', 'R', 'L', 'L', 'L', 'F', 'L', 'L', 'L', 'U', 'D', 'D']
    print(filter_records(record))