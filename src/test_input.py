# ========== 测试输入模拟（本地调试时使用）==========
test_input = [
    "0 1 1 2"         # 你可以改这里的测试数据
]
input_iter = iter(test_input)
original_input = input
input = lambda: next(input_iter)
# ==================================================

# 正式代码开始
array = list(map(int, input().split()))

# 如果输入为空，直接输出0
if not array:
    print(0)
    exit()

max_val = max(array)
cnt = [0] * (max_val + 1)

for x in array:
    cnt[x] += 1

ans = cnt[0]        # 初始面积
used = cnt[0]       # 已处理箱子数

for k in range(1, len(cnt)):
    space = ans * (k + 1) - used
    if cnt[k] <= space:
        used += cnt[k]
    else:
        r = cnt[k] - space
        add = (r + k) // (k + 1)   # 向上取整
        ans += add
        used += cnt[k]

print(ans)