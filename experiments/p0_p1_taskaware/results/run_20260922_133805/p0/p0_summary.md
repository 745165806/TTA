# P0 mechanism diagnosis summary

**POST_HOC_DEVELOPMENT_ONLY**: target10 是 development/diagnostic set，不是 untouched test。

Frozen (K=0) EER = 0.098571, AUC = 0.963309; threshold tau0 = -4.770049

## 1. view variance 降低是否对应 EER/AUC 改善？

view_reduction vs -EER Spearman = 0.838, view_reduction vs AUC Spearman = -0.747. EER 只有微弱正向秩相关（且 EER 只取两个几乎相同的值，属工作点噪声），而 AUC 随 view_reduction 单调恶化 —— 不存在真正的 EER/AUC 双重改善。

## 2. EP 的平均 signed task delta 是正、零还是负？

overall mean signed task delta = -0.005681 -> negative（UPDATE_DIRECTION_NOT_TASK_ALIGNED）

## 3. 更新对 bonafide 和 spoof 是否存在明显不对称？

bonafide mean signed delta = 0.000481, spoof mean signed delta = -0.016563。存在明显不对称：spoof 样本被明显推向错误方向（负向更大）。

## 4. helpful flip 是否多于 harmful flip？

helpful = 1, harmful = 2, net = -1。分数位移量级过小，几乎不产生决策翻转。

## 5. guard/revert 是否与性能恶化相关？

guard_activation_rate vs -EER = 0.860（但 vs AUC = -0.774，激活越多 AUC 越差）；guard_revert_rate vs -EER = 0.395。guard 激活与 AUC 恶化相关。

## 6. view1/view2 是否明显弱于 original view？

view0 EER=0.098571 AUC=0.963309; view1 EER=0.139252 AUC=0.933728; view2 EER=0.099064 AUC=0.963218。view1（噪声增强）明显更弱；view2（FIR）几乎与 original 持平。

## 7. 是否有证据支持停止继续优化纯 view_variance？

是：view_reduction 增大时 AUC 单调恶化、mean signed task delta 全为负、EER 只有工作点级噪声改善。停止继续优化纯 view_variance（OBJECTIVE_OPTIMIZED_BUT_NO_TASK_GAIN）。
