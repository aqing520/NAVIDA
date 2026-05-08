import json
import re

path = "data/train_r2r_rxr_qwen35_full.jsonl"

def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def is_probable_leak(user, ans):
    user_n = norm(user)
    ans_n = norm(ans)

    # 单独的 stop 太短，VLN 指令里自然会出现 stop，不判泄漏
    if ans_n == "stop":
        return False

    # 太短的答案也不做 exact leak
    if len(ans_n.split()) <= 2:
        return False

    # 完整 action chunk 出现在 user 里，才算高危
    if ans_n in user_n:
        return True

    return False

leaks = []
total = 0
stop_ans = 0
stop_ans_user_has_stop = 0

with open(path) as f:
    for idx, line in enumerate(f):
        ex = json.loads(line)
        total += 1
        user = ex["conversations"][0]["value"]
        ans = ex["conversations"][1]["value"]

        if norm(ans) == "stop":
            stop_ans += 1
            if "stop" in norm(user):
                stop_ans_user_has_stop += 1

        if is_probable_leak(user, ans):
            leaks.append((idx, ex.get("task type"), ex.get("id"), ans, user[-800:]))
            if len(leaks) >= 20:
                break

print("total:", total)
print("stop answer samples:", stop_ans)
print("stop answer and user contains stop:", stop_ans_user_has_stop)
print("probable leaks:", len(leaks))

for item in leaks:
    print("\nLEAK?", item[0], item[1], item[2])
    print("answer:", item[3])
    print("user_tail:", item[4])