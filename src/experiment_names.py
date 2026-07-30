"""Canonical experiment identifiers and paper-facing names."""

EXPERIMENT_NAMES = {
    "A": "GBR传统机器学习",
    "A_prime": "ANN传统神经网络",
    "B": "传统联邦学习",
    "C": "多智能体协同联邦学习",
    "C_bias_corrected": "多智能体协同联邦学习（偏差校正）",
    "FEDYOGI": "自适应联邦学习（FedYogi-TR）",
    "FEDYOGI_bias_corrected": "自适应联邦学习（FedYogi-TR）（偏差校正）",
    "MAS_ADAPTIVE": "多智能体协同自适应联邦学习（FedYogi-TR）",
    "MAS_ADAPTIVE_bias_corrected": "多智能体协同自适应联邦学习（FedYogi-TR）（偏差校正）",
    "VG_FEDYOGI_TR": "验证引导自适应联邦学习（VG-FedYogi-TR）",
    "VG_FEDYOGI_TR_bias_corrected": "验证引导自适应联邦学习（VG-FedYogi-TR）（偏差校正）",
    "MAS_VG_FEDYOGI_TR": "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）",
    "MAS_VG_FEDYOGI_TR_bias_corrected": "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）（偏差校正）",
    "COHERENCE_FEDYOGI_TR": "一致性感知自适应联邦学习（Coherence-FedYogi-TR）",
    "COHERENCE_FEDYOGI_TR_bias_corrected": "一致性感知自适应联邦学习（Coherence-FedYogi-TR）（偏差校正）",
    "LLM_GCA_FEDYOGI_TR": "大模型引导的生成式一致性感知自适应联邦学习（LLM-GCA-FedYogi-TR）",
    "LLM_GCA_FEDYOGI_TR_bias_corrected": "大模型引导的生成式一致性感知自适应联邦学习（LLM-GCA-FedYogi-TR）（偏差校正）",
    "STRICT_COHERENCE_FEDYOGI_TR": "严格无服务器数据一致性感知自适应联邦学习（Strict-Coherence-FedYogi-TR）",
    "STRICT_COHERENCE_FEDYOGI_TR_bias_corrected": "严格无服务器数据一致性感知自适应联邦学习（Strict-Coherence-FedYogi-TR）（偏差校正）",
    "LLM_STRICT_GCA_FEDYOGI_TR": "严格无服务器数据大模型一致性感知自适应联邦学习（LLM-Strict-GCA-FedYogi-TR）",
    "LLM_STRICT_GCA_FEDYOGI_TR_bias_corrected": "严格无服务器数据大模型一致性感知自适应联邦学习（LLM-Strict-GCA-FedYogi-TR）（偏差校正）",
    "VP_GCA_FEDYOGI_TR": "Validation-preview generative coherence-aware FedYogi-TR",
    "VP_GCA_FEDYOGI_TR_bias_corrected": "Validation-preview generative coherence-aware FedYogi-TR (bias corrected)",
    "LLM_VP_GCA_FEDYOGI_TR": "LLM-guided validation-preview generative FedYogi-TR",
    "LLM_VP_GCA_FEDYOGI_TR_bias_corrected": "LLM-guided validation-preview generative FedYogi-TR (bias corrected)",
}

ABLATION_NAMES = {
    "ab-1": "传统联邦学习（FedAvg）",
    "ab-2": "自适应联邦学习（FedYogi-TR）",
    "ab-3": "验证引导自适应联邦学习（VG-FedYogi-TR）",
    "ab-4": "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）",
}

EXPERIMENT_ORDER = [
    "A",
    "A_prime",
    "B",
    "FEDYOGI",
    "VG_FEDYOGI_TR",
    "MAS_VG_FEDYOGI_TR",
    "COHERENCE_FEDYOGI_TR",
    "LLM_GCA_FEDYOGI_TR",
    "STRICT_COHERENCE_FEDYOGI_TR",
    "LLM_STRICT_GCA_FEDYOGI_TR",
    "VP_GCA_FEDYOGI_TR",
    "LLM_VP_GCA_FEDYOGI_TR",
]

LEGACY_EXPERIMENT_ORDER = ["C", "MAS_ADAPTIVE"]

EXPERIMENT_NAMES.update({
    "B_STRICT": "严格无服务器数据传统联邦学习（FedAvg-Final）",
    "B_STRICT_bias_corrected": "严格无服务器数据传统联邦学习（FedAvg-Final）（偏差校正）",
    "FEDYOGI_STRICT": "严格无服务器数据自适应联邦学习（FedYogi-TR-Final）",
    "FEDYOGI_STRICT_bias_corrected": "严格无服务器数据自适应联邦学习（FedYogi-TR-Final）（偏差校正）",
})

EXPERIMENT_ORDER.insert(EXPERIMENT_ORDER.index("FEDYOGI"), "B_STRICT")
EXPERIMENT_ORDER.insert(EXPERIMENT_ORDER.index("VG_FEDYOGI_TR"), "FEDYOGI_STRICT")

ALIASES = {
    "A": "A",
    "A_Centralized_GBR": "A",
    "A (GBR)": "A",
    "GBR传统机器学习": "A",
    "A_prime": "A_prime",
    "A_Prime_Neural_Network": "A_prime",
    "A' (NN)": "A_prime",
    "ANN传统神经网络": "A_prime",
    "B": "B",
    "B_FedAvg": "B",
    "B (FedAvg)": "B",
    "传统联邦学习": "B",
    "传统联邦学习（FedAvg）": "B",
    "C": "C",
    "C_MAS_FL_LLM": "C",
    "C (MAS-FL-LLM)": "C",
    "C + Bias Corr.": "C_bias_corrected",
    "FEDYOGI": "FEDYOGI",
    "FedYogi": "FEDYOGI",
    "FedYogi-TR": "FEDYOGI",
    "自适应联邦学习（FedYogi）": "FEDYOGI",
    "自适应联邦学习（FedYogi-TR）": "FEDYOGI",
    "FEDYOGI_bias_corrected": "FEDYOGI_bias_corrected",
    "自适应联邦学习（FedYogi-TR）（偏差校正）": "FEDYOGI_bias_corrected",
    "MAS_ADAPTIVE": "MAS_ADAPTIVE",
    "MAS Adaptive": "MAS_ADAPTIVE",
    "MAS_ADAPTIVE_bias_corrected": "MAS_ADAPTIVE_bias_corrected",
    "VG_FEDYOGI_TR": "VG_FEDYOGI_TR",
    "VG-FedYogi-TR": "VG_FEDYOGI_TR",
    "验证引导自适应联邦学习（VG-FedYogi-TR）": "VG_FEDYOGI_TR",
    "VG_FEDYOGI_TR_bias_corrected": "VG_FEDYOGI_TR_bias_corrected",
    "验证引导自适应联邦学习（VG-FedYogi-TR）（偏差校正）": "VG_FEDYOGI_TR_bias_corrected",
    "MAS_VG_FEDYOGI_TR": "MAS_VG_FEDYOGI_TR",
    "MAS-VG-FedYogi-TR": "MAS_VG_FEDYOGI_TR",
    "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）": "MAS_VG_FEDYOGI_TR",
    "MAS_VG_FEDYOGI_TR_bias_corrected": "MAS_VG_FEDYOGI_TR_bias_corrected",
    "多智能体协同验证引导自适应联邦学习（MAS-VG-FedYogi-TR）（偏差校正）": "MAS_VG_FEDYOGI_TR_bias_corrected",
    "COHERENCE_FEDYOGI_TR": "COHERENCE_FEDYOGI_TR",
    "Coherence-FedYogi-TR": "COHERENCE_FEDYOGI_TR",
    "Coherence-Aware FedYogi-TR": "COHERENCE_FEDYOGI_TR",
    "COHERENCE_FEDYOGI_TR_bias_corrected": "COHERENCE_FEDYOGI_TR_bias_corrected",
    "LLM_GCA_FEDYOGI_TR": "LLM_GCA_FEDYOGI_TR",
    "LLM-GCA-FedYogi-TR": "LLM_GCA_FEDYOGI_TR",
    "LLM_GCA_FEDYOGI_TR_bias_corrected": "LLM_GCA_FEDYOGI_TR_bias_corrected",
    "STRICT_COHERENCE_FEDYOGI_TR": "STRICT_COHERENCE_FEDYOGI_TR",
    "Strict-Coherence-FedYogi-TR": "STRICT_COHERENCE_FEDYOGI_TR",
    "STRICT_COHERENCE_FEDYOGI_TR_bias_corrected": "STRICT_COHERENCE_FEDYOGI_TR_bias_corrected",
    "LLM_STRICT_GCA_FEDYOGI_TR": "LLM_STRICT_GCA_FEDYOGI_TR",
    "LLM-Strict-GCA-FedYogi-TR": "LLM_STRICT_GCA_FEDYOGI_TR",
    "LLM_STRICT_GCA_FEDYOGI_TR_bias_corrected": "LLM_STRICT_GCA_FEDYOGI_TR_bias_corrected",
    "VP_GCA_FEDYOGI_TR": "VP_GCA_FEDYOGI_TR",
    "VP-GCA-FedYogi-TR": "VP_GCA_FEDYOGI_TR",
    "VP_GCA_FEDYOGI_TR_bias_corrected": "VP_GCA_FEDYOGI_TR_bias_corrected",
    "LLM_VP_GCA_FEDYOGI_TR": "LLM_VP_GCA_FEDYOGI_TR",
    "LLM-VP-GCA-FedYogi-TR": "LLM_VP_GCA_FEDYOGI_TR",
    "LLM_VP_GCA_FEDYOGI_TR_bias_corrected": "LLM_VP_GCA_FEDYOGI_TR_bias_corrected",
}


def canonical_experiment_key(value: str) -> str:
    """Return the stable internal key for a legacy key or display name."""
    return ALIASES.get(str(value), str(value))


def experiment_display_name(value: str) -> str:
    """Return the paper-facing semantic experiment name."""
    key = canonical_experiment_key(value)
    return EXPERIMENT_NAMES.get(key, str(value))
