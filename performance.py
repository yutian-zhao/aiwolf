import numpy as np

GAME_TYPE = 15
def get_content_list(content):
    content_list = []
    begin_idx = None
    lb = 0
    for idx, s in enumerate(content):
        if s == '[':
            lb += 1
            if begin_idx is None:
                begin_idx = idx
        if s == ']':
            lb -= 1
            if begin_idx is not None and lb==0:
                content_list.append(content[begin_idx:idx+1])
                begin_idx = None
    return content_list

def parse_array(s):
    
    if s.startswith('[['):
        arr = []
        array_list = get_content_list(s[1:-1])
        for a in array_list:
            arr.append(parse_array(a))
        return arr
    elif s.startswith('['):
        arr = []
        array_list = s[1:-1].split(', ')
        
        for a in array_list:
            arr.append(parse_array(a))
        return arr
    else:
        return float(s)


def extract_data(lines):
    onnx_preds = []
    prob_lists = []
    # executed = []
    result = []
    days = 0
    for line in lines:
        line = line.strip()
        if line.startswith("YUTIAN") and "result" in line:
            start_idx = line.find("[")
            res = parse_array(line[start_idx:])
            for i in range(days):
                result.append(res)
            days = 0
        # if line.startswith("YUTIAN") and "Execute" in line:
        #     start_idx = line.find("is ")
        #     end_idx = line.find("(need")
        #     executed_idx = int(line[start_idx+3:end_idx])-1
        #     for i in range(len(onnx_preds)-len(executed)):
        #         executed.append(executed_idx)
        if line.startswith("YUTIAN") and '[[' in line:
            start_idx = line.find("[[")
            if 'onnx' in line:
                days += 1
                onnx_preds.append(parse_array(line[start_idx:]))
            elif 'getProb' in line:
                prob_lists.append(parse_array(line[start_idx:]))
    return np.array(onnx_preds), np.array(prob_lists), np.array(result) #, np.array(executed)

def accuracy(input, target):
    # input B C D; target B D
    # output C
    result = np.zeros(input.shape[0], input.shape[1])
    for b in range(input.shape[0]):
        for c in range(input.shape[1]):
            result[b][c] = (target[b][input[b][c]]==c).float()
    return result

fname = "result_yutian_15.log"
with open(fname, 'r') as f:
    lines = f.readlines()
    onnx_preds, prob_lists, result = extract_data(lines)
# print(len(onnx_preds))
result -= 1
if GAME_TYPE==15:
    # probs w, v, s, p, m, b
    # onnx v, s, m, b, w, p
    prob_lists = prob_lists[:, [1, 2, 4, 5, 0, 3], :]
onnx_pred = np.argmax(onnx_preds[:, :6, :], axis=-1) # B C 
prob_list = np.argmax(prob_lists, axis=-1)
onnx_acc = accuracy(onnx_pred, result)
prob_acc = accuracy(prob_list, result)
onnx_data = np.add.accumulate(np.equal(onnx_acc, result), axis=0)
prob_data = np.add.accumulate(np.equal(prob_list, result), axis=0)


# print(onnx_preds.shape, prob_lists.shape, result.shape)
# print(np.sum(onnx_preds, axis=-2))
# s = '[1, 3, 2]'
# print(parse_array(s))
# lines = ["YUTIAN result: [1, 5, 1, 1, 6, 1, 5, 1, 3, 5, 2, 1, 1, 4, 1]"]
# _, _, a = extract_data(lines)
# print(a)
# ss= '[1, 5, 1, 1, 6, 1, 5, 1, 3, 5, 2, 1, 1, 4, 1]'
# print(parse_array(ss))