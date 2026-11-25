#!/bin/bash

# 设置 Conda 环境名
CONDA_ENV="LLMenv"

# 设置 Python 文件路径
PYTHON_FILE="./main.py"

# 设置其他参数
WEIGHT_GLOBAL=1.0
WEIGHT_FINALCF=0.2

WEIGHT_INTER=1.0
WEIGHT_EMB_D=0.7
WEIGHT_CATE_R=2.0
TAU_OUT=1.0
N_CATE=5
WEIGHT_LAMBDA=4

#WEIGHT_INTER=1.0
#WEIGHT_CATE_D=0.5
#WEIGHT_EMB_D=0.7
#WEIGHT_CATE_R=0.1
#TAU_OUT=1.0

#SEED=123  # 新的参数 seed，默认值为 456
#SEED=2040  # 新的参数 seed，默认值为 456

# 激活 Conda 环境
echo "正在激活 Conda 环境: $CONDA_ENV..."
. $(conda info --base)/etc/profile.d/conda.sh
conda activate "$CONDA_ENV"

# 检查 Conda 环本是否成功激活
if [ $? -ne 0 ]; then
    echo "激活 Conda 环境 $CONDA_ENV 失败！"
    exit 1
fi

# 检查 Python 文件是否存在
if [ ! -f "$PYTHON_FILE" ]; then
    echo "Python 文件 $PYTHON_FILE 不存在！"
    exit 1
fi

# 无限循环执行 Python 程序
#for i in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
for i in 123 456; do

#for i in 0.6 0.7 0.8 0.9; do
#for i in  0.9 1.0; do
#for i in 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do

#for i in 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do
#for i in 1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do

#for i in 10.0 9.0 8.0 7.0 ; do
#for i in 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0 10.0; do

#for i in 9.0 10.0; do

#for i in 1.0; do
#for i in 0.1 0.2 0.3 0.4 0.5; do

    # 跳过 WEIGHT_FINALCF 为 4 或 8 的情况
#    if [ "$(echo "$i == 1.0" | bc)" -eq 1 ] ; then
##    if [ $i -eq 1 ] || [ $i -eq 8 ]; then
#        echo "跳过 WEIGHT_CATE_D=$i 的情况"
#        continue
#    fi

    # 设置当前的 weight_FinalCF
    SEED=$i  # 每次循环时将 WEIGHT_FINALCF 设置为当前的 i 值

    # 构造文件名，文件名包括 myout1223-domain3-1202woS-F 和 weight_FinalCF 的值
#     myout1227-domain3-0102-C2-F4-llre3-seed456
#  中心和warmemb全部detach
#    OUTPUT_FILE="myout0129-warmEmbDetach-domain3--S0.1C${WEIGHT_CATE_D}E${WEIGHT_EMB_D}R${WEIGHT_CATE_R}-F${WEIGHT_FINALCF}T${TAU_OUT}-I1-NEG128-seed${SEED}"
#    OUTPUT_FILE="myout0204WOMix-domain1-L${WEIGHT_LAMBDA}-S0.1E${WEIGHT_EMB_D}R${WEIGHT_CATE_R}-F${WEIGHT_FINALCF}T${TAU_OUT}-CATE${N_CATE}-seed${SEED}"
    OUTPUT_FILE="myout0518-domain0-L${WEIGHT_LAMBDA}-S0.1E${WEIGHT_EMB_D}R${WEIGHT_CATE_R}-F${WEIGHT_FINALCF}T${TAU_OUT}-CATE${N_CATE}-seed${SEED}"

    # 打印出参数值，确保我们可以实时看到这些值
    echo "正在运行 Python 程序，当前参数："
    echo "WEIGHT_GLOBAL=${WEIGHT_GLOBAL}"
    echo "WEIGHT_FINALCF=${WEIGHT_FINALCF}"
    echo "WEIGHT_EMB_D=${WEIGHT_EMB_D}"
    echo "WEIGHT_CATE_R=${WEIGHT_CATE_R}"

    echo "SEED=${SEED}"

    # 使用 nohup 运行 Python 程序并将输出重定向到动态生成的文件
    echo "正在后台运行 Python 程序 (WEIGHT_FINALCF=${WEIGHT_FINALCF})..."
    nohup python "$PYTHON_FILE" --n_cate "$N_CATE" --weight_lambda "$WEIGHT_LAMBDA" --weight_inter "$WEIGHT_INTER" --weight_emb_D "$WEIGHT_EMB_D" --weight_cate_R "$WEIGHT_CATE_R" --weight_global "$WEIGHT_GLOBAL" --weight_FinalCF "$WEIGHT_FINALCF" --tau_out "$TAU_OUT"  --seed "$SEED" > "$OUTPUT_FILE" 2>&1 &

    # 获取 Python 程序的进程 ID (PID)
    PID=$!

    # 启动一个新的终端进程来实时显示输出日志，同时显示当前的参数
    echo "实时显示日志：$OUTPUT_FILE"
    tail -f "$OUTPUT_FILE" &

    # 等待 Python 程序执行完成
    wait $PID

    # 检查 Python 程序是否成功完成
    if [ $? -eq 0 ]; then
        echo "Python 程序已成功完成，正在准备下一次执行..."
    else
        echo "Python 程序执行失败！"
        exit 1
    fi

    # 可选择添加暂停时间（如果需要）
    # sleep 10  # 例如等待10秒后重新执行
done

# 退出 Conda 环境
conda deactivate

echo "脚本执行完毕！"