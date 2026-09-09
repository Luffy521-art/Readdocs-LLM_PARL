<img width="415" height="226" alt="image" src="https://github.com/user-attachments/assets/884a4aee-e343-40ec-83a6-4b68a43f970d" />

<img width="764" height="1038" alt="image" src="https://github.com/user-attachments/assets/048a5b77-2d1a-45d5-8739-cbd3e9e32e78" />

<img width="416" height="448" alt="image" src="https://github.com/user-attachments/assets/759af5c8-c7f2-4c8f-8a00-815dd767a314" />

初始化策略模型参数 θ、价值模型参数 φ、奖励模型参数 ψ

对每个外循环轮次 k = 1, 2, ... 执行：

读取当前训练批次的状态：
    q_all = [q_1, q_2, ..., q_B]^T

对当前批次执行 U 次 PPO 内循环：
    对 q_b，b = 1, 2, ..., B 执行：
        使用当前策略模型进行 rollout
        生成 response_b
        记录生成序列 sequences_b
        记录动作掩码 action_mask_b
        记录对数概率 log pi_θ(a_b | q_b)

    组装当前一次 rollout 的完整样本：
        D = {(q_b, response_b, sequences_b, action_mask_b)}_{b=1}^B

    对每个样本计算稠密奖励：
        对 sequences_b 做 teacher forcing
        提取 hidden_states_b
        将 hidden_states_b 输入奖励模型
        得到 token 级奖励 r_b

    根据 action_mask_b 对奖励进行对齐与截断：
        得到 reward_b

    构造 PPO 训练数据：
        training_data = {(D_b, reward_b)}_{b=1}^B

    执行 PPO 更新：
        θ, φ ← PPO_Update(θ, φ, training_data)

重新采样 PARL 子批次：
    选择一小部分训练样本 q_parl
    使用最新策略模型重新 rollout
    通过外部约束评分器得到监督标签 y_parl

执行 PARL 隐式梯度更新：
    对 q_parl 做 teacher forcing
    提取 hidden_states_parl
    计算直接监督损失：
        L_direct = MSE(R_ψ(hidden_states_parl), y_parl)

    对 L_direct 进行反向传播并保留计算图

    基于策略输出和 action_mask 构造策略侧梯度方向
    使用共轭梯度法近似求解二阶系统
    得到隐式修正向量 v

    计算 reward_model 参数的隐式梯度修正项
    叠加 L2 正则梯度：
        L_reg = λ_l2 ||ψ||^2

    执行奖励模型参数更新：
        ψ ← ψ - β · ∇ψ (L_direct + 隐式修正项 + L_reg)

使用更新后的奖励模型重新打分当前样本
记录训练日志、loss 和奖励统计量
