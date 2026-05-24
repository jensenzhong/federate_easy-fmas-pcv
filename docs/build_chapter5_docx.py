from __future__ import annotations

import datetime as dt
import struct
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd


EMU_PER_INCH = 914400
MAX_IMAGE_WIDTH_EMU = int(5.9 * EMU_PER_INCH)


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as f:
        header = f.read(24)
    if header[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"Unsupported image format: {path}")
    width, height = struct.unpack(">II", header[16:24])
    return width, height


def xml_text(text: str) -> str:
    return escape(text).replace("\n", "&#10;")


def run_text(text: str, bold: bool = False, size: int = 24, east_asia: str = "宋体") -> str:
    rpr = (
        "<w:rPr>"
        '<w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" '
        f'w:eastAsia="{east_asia}"/>'
        f'<w:sz w:val="{size}"/>'
        f'<w:szCs w:val="{size}"/>'
        f"{'<w:b/>' if bold else ''}"
        "</w:rPr>"
    )
    return f'<w:r>{rpr}<w:t xml:space="preserve">{xml_text(text)}</w:t></w:r>'


def paragraph(
    text: str,
    style: str = "Normal",
    align: str | None = None,
    bold: bool = False,
) -> str:
    jc = f'<w:jc w:val="{align}"/>' if align else ""
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/>{jc}</w:pPr>'
    return f"<w:p>{ppr}{run_text(text, bold=bold)}</w:p>"


def image_paragraph(rid: str, cx: int, cy: int, docpr_id: int, name: str) -> str:
    drawing = f"""
    <w:r>
      <w:drawing>
        <wp:inline distT="0" distB="0" distL="0" distR="0"
          xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
          xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"
          xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">
          <wp:extent cx="{cx}" cy="{cy}"/>
          <wp:docPr id="{docpr_id}" name="{xml_text(name)}"/>
          <wp:cNvGraphicFramePr>
            <a:graphicFrameLocks noChangeAspect="1"/>
          </wp:cNvGraphicFramePr>
          <a:graphic>
            <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
              <pic:pic>
                <pic:nvPicPr>
                  <pic:cNvPr id="{docpr_id}" name="{xml_text(name)}"/>
                  <pic:cNvPicPr/>
                </pic:nvPicPr>
                <pic:blipFill>
                  <a:blip r:embed="{rid}"/>
                  <a:stretch><a:fillRect/></a:stretch>
                </pic:blipFill>
                <pic:spPr>
                  <a:xfrm>
                    <a:off x="0" y="0"/>
                    <a:ext cx="{cx}" cy="{cy}"/>
                  </a:xfrm>
                  <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
                </pic:spPr>
              </pic:pic>
            </a:graphicData>
          </a:graphic>
        </wp:inline>
      </w:drawing>
    </w:r>
    """
    return f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{drawing}</w:p>'


def table(rows: list[list[str]]) -> str:
    tbl_pr = """
    <w:tblPr>
      <w:tblW w:w="0" w:type="auto"/>
      <w:tblBorders>
        <w:top w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:left w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:bottom w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:right w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:insideH w:val="single" w:sz="8" w:space="0" w:color="000000"/>
        <w:insideV w:val="single" w:sz="8" w:space="0" w:color="000000"/>
      </w:tblBorders>
    </w:tblPr>
    """
    grid = "<w:tblGrid>" + "".join('<w:gridCol w:w="1700"/>' for _ in rows[0]) + "</w:tblGrid>"
    trs = []
    for r_i, row in enumerate(rows):
        tcs = []
        for cell in row:
            cell_bold = r_i == 0
            tcs.append(
                "<w:tc>"
                '<w:tcPr><w:tcW w:w="1700" w:type="dxa"/></w:tcPr>'
                f'<w:p><w:pPr><w:jc w:val="center"/></w:pPr>{run_text(cell, bold=cell_bold)}</w:p>'
                "</w:tc>"
            )
        trs.append("<w:tr>" + "".join(tcs) + "</w:tr>")
    return "<w:tbl>" + tbl_pr + grid + "".join(trs) + "</w:tbl>"


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
        <w:sz w:val="24"/>
        <w:szCs w:val="24"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:line="420" w:lineRule="auto" w:after="120"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Title">
    <w:name w:val="Title"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="120" w:after="240"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:b/>
      <w:sz w:val="32"/>
      <w:szCs w:val="32"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:b/>
      <w:sz w:val="28"/>
      <w:szCs w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="caption"/>
    <w:basedOn w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="60" w:after="120"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
      <w:sz w:val="21"/>
      <w:szCs w:val="21"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def settings_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:settings xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:zoom w:percent="100"/>
</w:settings>
"""


def content_types_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Default Extension="png" ContentType="image/png"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
  <Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
  <Override PartName="/word/settings.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.settings+xml"/>
  <Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
  <Override PartName="/docProps/app.xml" ContentType="application/vnd.openxmlformats-officedocument.extended-properties+xml"/>
</Types>
"""


def package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/extended-properties" Target="docProps/app.xml"/>
</Relationships>
"""


def core_xml() -> str:
    created = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties"
 xmlns:dc="http://purl.org/dc/elements/1.1/"
 xmlns:dcterms="http://purl.org/dc/terms/"
 xmlns:dcmitype="http://purl.org/dc/dcmitype/"
 xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <dc:title>第五章 结果与讨论</dc:title>
  <dc:creator>OpenAI Codex</dc:creator>
  <cp:lastModifiedBy>OpenAI Codex</cp:lastModifiedBy>
  <dcterms:created xsi:type="dcterms:W3CDTF">{created}</dcterms:created>
  <dcterms:modified xsi:type="dcterms:W3CDTF">{created}</dcterms:modified>
</cp:coreProperties>
"""


def app_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
 xmlns:vt="http://schemas.openxmlformats.org/officeDocument/2006/docPropsVTypes">
  <Application>Microsoft Office Word</Application>
  <DocSecurity>0</DocSecurity>
  <ScaleCrop>false</ScaleCrop>
  <Company></Company>
  <LinksUpToDate>false</LinksUpToDate>
  <SharedDoc>false</SharedDoc>
  <HyperlinksChanged>false</HyperlinksChanged>
  <AppVersion>16.0000</AppVersion>
</Properties>
"""


def fmt_pct(value: float) -> str:
    return f"{value:.2f}%"


def fmt_int(value: float) -> str:
    return f"{int(round(value)):,}"


def fig_block(blocks: list[str], image_name: str, rid: str, docpr_id: int, caption: str, width_inch: float = 5.9) -> None:
    img = Path.cwd() / "ch5_figures" / image_name
    w, h = png_size(img)
    cx = int(width_inch * EMU_PER_INCH)
    cy = int(cx * h / w)
    blocks.append(image_paragraph(rid, cx, cy, docpr_id, image_name))
    blocks.append(paragraph(caption, style="Caption"))


def build_docx(output_path: Path) -> None:
    root = Path.cwd().parent

    central = pd.read_csv(root / "results" / "centralized_results.csv").iloc[0]
    nn = pd.read_csv(root / "results" / "centralized_nn_results.csv").iloc[0]
    fed = pd.read_csv(root / "results" / "fedavg_results.csv").iloc[0]
    mas = pd.read_csv(root / "results" / "scenario_c_results.csv").iloc[0]
    round_b = pd.read_csv(root / "results" / "logs" / "scene_B_round_metrics.csv")
    round_c = pd.read_csv(root / "results" / "logs" / "scene_C_round_metrics.csv")
    ablation = pd.read_csv(root / "results" / "ablation_summary.csv")
    multi = pd.read_csv(root / "results" / "multi_seed" / "all_results.csv")

    best_b = round_b.loc[round_b["global_val_mape"].idxmin()]
    best_c = round_c.loc[round_c["global_val_mape"].idxmin()]
    b_vals = multi.loc[multi["scenario"] == "B", "test_mape"] * 100
    c_vals = multi.loc[multi["scenario"] == "C", "test_mape"] * 100

    improve_vs_fed_rel = (fed["test_mape"] - mas["test_mape"]) / fed["test_mape"] * 100
    improve_vs_fed_abs = (fed["test_mape"] - mas["test_mape"]) * 100
    improve_vs_nn_rel = (nn["test_mape"] - mas["test_mape"]) / nn["test_mape"] * 100
    improve_vs_central_rel = (central["test_mape"] - mas["test_mape"]) / central["test_mape"] * 100

    rmse_improve = (mas["test_rmse"] - mas["test_rmse_corrected"]) / mas["test_rmse"] * 100
    mae_improve = (mas["test_mae"] - mas["test_mae_corrected"]) / mas["test_mae"] * 100
    mpe_abs_improve = (abs(mas["test_mpe"]) - abs(mas["test_mpe_corrected"])) / abs(mas["test_mpe"]) * 100

    strategy_counts = round_c["strategy"].value_counts().to_dict()
    fairness_ratio = strategy_counts.get("fairness_clip", 0) / len(round_c) * 100

    b_gap = (
        max(best_b["Client 1_val_mape"], best_b["Client 2_val_mape"], best_b["Client 3_val_mape"])
        - min(best_b["Client 1_val_mape"], best_b["Client 2_val_mape"], best_b["Client 3_val_mape"])
    ) * 100
    c_gap = (
        max(best_c["Client 1_val_mape"], best_c["Client 2_val_mape"], best_c["Client 3_val_mape"])
        - min(best_c["Client 1_val_mape"], best_c["Client 2_val_mape"], best_c["Client 3_val_mape"])
    ) * 100

    main_table = [
        ["方法", "平均绝对百分比误差", "均方根误差", "平均绝对误差", "平均百分比误差", "决定系数"],
        ["集中式梯度提升树基线", fmt_pct(central["test_mape"] * 100), fmt_int(central["test_rmse"]), fmt_int(central["test_mae"]), fmt_pct(central["test_mpe"] * 100), "-"],
        ["集中式多层感知机基线", fmt_pct(nn["test_mape"] * 100), fmt_int(nn["test_rmse"]), fmt_int(nn["test_mae"]), fmt_pct(nn["test_mpe"] * 100), "-"],
        ["固定权重联邦训练基线", fmt_pct(fed["test_mape"] * 100), fmt_int(fed["test_rmse"]), fmt_int(fed["test_mae"]), fmt_pct(fed["test_mpe"] * 100), f"{fed['test_r2']:.4f}"],
        ["基于多智能体协同的联邦学习训练", fmt_pct(mas["test_mape"] * 100), fmt_int(mas["test_rmse"]), fmt_int(mas["test_mae"]), fmt_pct(mas["test_mpe"] * 100), f"{mas['test_r2']:.4f}"],
    ]

    seed_table = [
        ["方法", "平均值", "标准差", "最优值", "最差值"],
        ["固定权重联邦训练基线", fmt_pct(b_vals.mean()), fmt_pct(b_vals.std(ddof=1)), fmt_pct(b_vals.min()), fmt_pct(b_vals.max())],
        ["基于多智能体协同的联邦学习训练", fmt_pct(c_vals.mean()), fmt_pct(c_vals.std(ddof=1)), fmt_pct(c_vals.min()), fmt_pct(c_vals.max())],
    ]

    ablation_rows = []
    for _, row in ablation.iterrows():
        ablation_rows.append(
            (
                row["name"].replace("\\_", "_"),
                row["test_mape"] * 100,
                row["test_rmse"],
                row["test_mae"],
                row["test_mpe"] * 100,
                row["test_r2"],
            )
        )
    ablation_rows.sort(key=lambda x: x[1])
    best_ablation_name, best_ablation_mape, *_ = ablation_rows[0]

    blocks: list[str] = []
    blocks.append(paragraph("第五章 结果与讨论", style="Title"))
    blocks.append(
        paragraph(
            "本章在前文方法论的基础上，对基于多智能体协同的联邦学习训练进行系统化的结果报告与讨论。为保证比较结论具有可解释性，本文将其与集中式梯度提升树、集中式多层感知机以及固定权重联邦训练基线放在统一的数据与特征处理口径下进行对照。全部实验均基于清洗后的六百八十八条有效样本展开，其中训练集、验证集和测试集分别为五百五十条、六十八条和七十条；联邦场景包含三个客户端，其本地训练样本量分别为一百四十八条、一百五十条和一百四十条。由此，本章的差异主要来自训练机制与聚合方式，而不是数据划分口径差异。"
        )
    )

    blocks.append(paragraph("5.1 总体结果比较", style="Heading1"))
    blocks.append(
        paragraph(
            f"图5-1和表5-1展示了四类方法在测试集上的总体表现。以平均绝对百分比误差作为主评价指标时，基于多智能体协同的联邦学习训练取得了{fmt_pct(mas['test_mape'] * 100)}的结果，略优于固定权重联邦训练基线的{fmt_pct(fed['test_mape'] * 100)}，绝对差值为{improve_vs_fed_abs:.2f}个百分点，相对改善幅度为{improve_vs_fed_rel:.2f}%。这一提升幅度并不大，说明在统一二十轮训练设置下，两类联邦方案已经较为接近；但其改进方向是稳定一致的，因为均方根误差、平均绝对误差、平均百分比误差和决定系数也都较固定权重联邦训练基线略有改善。与此同时，该方法相较集中式多层感知机和集中式梯度提升树的相对改善幅度分别达到{improve_vs_nn_rel:.2f}%和{improve_vs_central_rel:.2f}%，说明在当前工程造价预测任务中，联邦建模框架本身仍然具有明显优势。"
        )
    )
    blocks.append(paragraph("表5-1  主要方法在测试集上的结果比较", style="Caption"))
    blocks.append(table(main_table))
    fig_block(blocks, "fig5_1_overall_comparison.png", "rId3", 1, "图5-1  不同方法在测试集上的总体表现")
    blocks.append(
        paragraph(
            f"从误差结构看，基于多智能体协同的联邦学习训练的平均百分比误差为{fmt_pct(mas['test_mpe'] * 100)}，固定权重联邦训练基线为{fmt_pct(fed['test_mpe'] * 100)}，二者都表现为系统性低估，但前者的低估幅度略小。这表明在当前结果下，多智能体协同带来的收益并不体现为对某一单一指标的激进压缩，而更接近一种对整体误差结构的细微优化。对工程造价预测而言，这种小幅但方向一致的改进是有意义的，因为模型不仅要追求数值更低，还要追求误差分布更加稳定、偏差方向更加可控。"
        )
    )

    blocks.append(paragraph("5.2 收敛过程与决策轨迹分析", style="Heading1"))
    blocks.append(
        paragraph(
            f"图5-2给出了固定权重联邦训练基线与基于多智能体协同的联邦学习训练在验证集上的收敛曲线。与前一版实验口径不同，本次比较中两者均采用二十轮训练，因此不再讨论通信轮次节省问题，而是直接比较相同轮数下的收敛质量。结果表明，固定权重联邦训练基线在第{int(best_b['round'])}轮达到最佳验证结果，平均绝对百分比误差为{fmt_pct(best_b['global_val_mape'] * 100)}；基于多智能体协同的联邦学习训练同样在第{int(best_c['round'])}轮达到最佳验证结果，平均绝对百分比误差为{fmt_pct(best_c['global_val_mape'] * 100)}。二者的最终验证表现极为接近，但基于多智能体协同的联邦学习训练仍然略优，这与其在测试集上的小幅领先是相互一致的。"
        )
    )
    fig_block(blocks, "fig5_2_convergence.png", "rId4", 2, "图5-2  两种联邦训练方式在验证集上的收敛过程比较")
    blocks.append(
        paragraph(
            f"图5-3进一步揭示了多智能体协同决策的演化轨迹。当前二十轮结果显示，系统在前四轮依次试探按样本量聚合、按性能聚合、混合聚合以及公平约束聚合，随后有{strategy_counts.get('fairness_clip', 0)}轮继续采用公平约束聚合，占总轮次的{fairness_ratio:.2f}%。学习率和本地训练轮数在该轮次配置下未再发生额外调节，说明多智能体协同机制在当前实验中并未表现为频繁切换规则，而是在早期完成探索后快速收敛到一种稳定策略。换言之，这一机制的价值不在于持续制造波动，而在于基于训练反馈选择并维持一个可解释的训练控制方案。"
        )
    )
    fig_block(blocks, "fig5_3_strategy_timeline.png", "rId5", 3, "图5-3  基于多智能体协同的联邦学习训练的决策轨迹")

    blocks.append(paragraph("5.3 客户端均衡性与消融结果分析", style="Heading1"))
    blocks.append(
        paragraph(
            f"图5-4展示了三个客户端验证误差的变化趋势。在最佳验证轮次处，基于多智能体协同的联邦学习训练下三个客户端的平均绝对百分比误差分别为{fmt_pct(best_c['Client 1_val_mape'] * 100)}、{fmt_pct(best_c['Client 2_val_mape'] * 100)}和{fmt_pct(best_c['Client 3_val_mape'] * 100)}，对应聚合权重为{best_c['Client 1_weight']:.3f}、{best_c['Client 2_weight']:.3f}和{best_c['Client 3_weight']:.3f}。与固定权重联邦训练基线相比，客户端间误差极差分别为{b_gap:.2f}个百分点和{c_gap:.2f}个百分点，差异非常有限。这说明基于多智能体协同的联邦学习训练并未通过牺牲某一个客户端的表现来换取总体指标，而是在相近的客户端均衡性水平上实现了轻微的整体改进。"
        )
    )
    fig_block(blocks, "fig5_4_client_trends.png", "rId6", 4, "图5-4  三个客户端验证误差的变化趋势")
    blocks.append(
        paragraph(
            f"图5-5对应的消融实验更能说明性能变化来自何处。首先，单独引入联邦近端约束后，平均绝对百分比误差由42.07%变为42.06%，几乎没有发生变化，说明近端约束在当前样本规模和异质性条件下更多承担训练稳定化作用，而不是主要增益来源。其次，当聚合规则从按样本量固定加权改为按性能固定加权时，误差降至41.53%，成为本组消融中最低值；改为固定混合加权后，误差为41.95%，同样优于按样本量固定加权。这表明，聚合规则设计本身是影响联邦训练质量的关键因素。再次，当前这版基于多智能体协同的联邦学习训练结果为42.01%，与最优固定策略相比并未继续降低主指标。这一现象需要结合图5-3理解：在本次二十轮训练中，多智能体协同机制很快收敛到公平约束聚合，表现出更强的保守性与约束性，因此其主要作用更接近于训练过程治理，而非单纯追求最低单点误差。就当前结果而言，本组消融中{best_ablation_name}对应的主指标最优，而多智能体协同机制的优势更多体现在提供可解释的策略选择路径和后续偏差校正空间。"
        )
    )
    fig_block(blocks, "fig5_5_ablation.png", "rId7", 5, "图5-5  不同组件组合下的消融结果比较")

    blocks.append(paragraph("5.4 稳健性与偏差校正分析", style="Heading1"))
    blocks.append(
        paragraph(
            f"为了判断该方法是否只是对单一随机种子敏感，图5-6和表5-2进一步比较了多随机种子条件下的结果分布。固定权重联邦训练基线的平均测试误差为{fmt_pct(b_vals.mean())}，标准差为{fmt_pct(b_vals.std(ddof=1))}；基于多智能体协同的联邦学习训练的平均测试误差为{fmt_pct(c_vals.mean())}，标准差为{fmt_pct(c_vals.std(ddof=1))}。从均值看，两者几乎一致；从最优值看，基于多智能体协同的联邦学习训练达到{fmt_pct(c_vals.min())}，优于固定权重联邦训练基线的{fmt_pct(b_vals.min())}；但从标准差看，其波动也更大。由此可见，该方法在当前数据规模下已经表现出更高的性能上界，但其平均收益尚未稳定转化为显著优势，因此仍需在更大样本和更多客户端条件下进一步验证。"
        )
    )
    blocks.append(paragraph("表5-2  多随机种子实验结果汇总", style="Caption"))
    blocks.append(table(seed_table))
    fig_block(blocks, "fig5_6_multiseed.png", "rId8", 6, "图5-6  多随机种子条件下的稳健性比较", width_inch=5.2)
    blocks.append(
        paragraph(
            f"图5-7展示了偏差校正前后的结果变化。对基于多智能体协同的联邦学习训练结果施加基于验证集偏差的比例校正后，平均百分比误差由{fmt_pct(mas['test_mpe'] * 100)}收敛到{fmt_pct(mas['test_mpe_corrected'] * 100)}，绝对偏差缩减了{mpe_abs_improve:.2f}%；均方根误差由{fmt_int(mas['test_rmse'])}下降到{fmt_int(mas['test_rmse_corrected'])}，降幅为{rmse_improve:.2f}%；平均绝对误差由{fmt_int(mas['test_mae'])}下降到{fmt_int(mas['test_mae_corrected'])}，降幅为{mae_improve:.2f}%；决定系数由{mas['test_r2']:.4f}提高到{mas['test_r2_corrected']:.4f}。需要指出的是，校正后的平均绝对百分比误差略微上升到{fmt_pct(mas['test_mape_corrected'] * 100)}。这说明偏差校正并非为了继续压低主指标，而是为了修正系统性低估，提高工程应用场景下的数值可靠性。从消融实验与偏差校正结果合并来看，多智能体协同训练的现实价值并不只体现在单次主指标最优，还体现在其后续可治理、可校正的空间。"
        )
    )
    fig_block(blocks, "fig5_7_bias_correction.png", "rId9", 7, "图5-7  偏差校正前后的结果变化")

    blocks.append(paragraph("5.5 本章讨论", style="Heading1"))
    blocks.append(
        paragraph(
            "综合以上结果，可以形成三点讨论。第一，在统一二十轮训练设置下，基于多智能体协同的联邦学习训练相较固定权重联邦训练基线的优势已经从过去较大的点位差距，转化为一种幅度较小但方向一致的综合改进。因此，对该方法的评价不宜建立在夸大性的领先叙述之上，而应建立在多指标共同改善这一更稳健的事实基础上。第二，消融实验表明，当前结果中最主要的性能贡献来自聚合规则本身，尤其是性能导向的固定加权策略；多智能体协同机制并未在本次重复实验中继续突破这一最优固定策略，而是更多体现为一种上层训练治理机制。第三，多随机种子实验和偏差校正实验共同说明，该方法的研究价值主要在于提供了一个可解释、可调节、可校正的联邦训练框架，它在最佳情形下具有较高性能上界，在工程应用中又可以通过后处理显著修正系统性偏差。"
        )
    )
    blocks.append(
        paragraph(
            "总体而言，本章结果支持这样一个更为审慎但更具说服力的判断：在高速公路工程造价预测这一具有数据分散与区域异质性特征的任务中，基于多智能体协同的联邦学习训练能够在与固定权重联邦训练基线相同的训练轮次下取得略优的综合表现，并为训练策略解释、客户端约束和偏差校正提供统一的分析框架；但在当前二十轮结果中，其主指标优势并不显著，且尚未稳定压过最优固定性能聚合策略。由此，本文更适合将其界定为一种具有明确研究价值和应用潜力的动态联邦训练机制，而不是在所有条件下均无条件占优的最终方案。"
        )
    )

    sect = """
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1800" w:header="708" w:footer="708" w:gutter="0"/>
    </w:sectPr>
    """

    document_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:wpc="http://schemas.microsoft.com/office/word/2010/wordprocessingCanvas" '
        'xmlns:mc="http://schemas.openxmlformats.org/markup-compatibility/2006" '
        'xmlns:o="urn:schemas-microsoft-com:office:office" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
        'xmlns:m="http://schemas.openxmlformats.org/officeDocument/2006/math" '
        'xmlns:v="urn:schemas-microsoft-com:vml" '
        'xmlns:wp14="http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing" '
        'xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing" '
        'xmlns:w10="urn:schemas-microsoft-com:office:word" '
        'xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" '
        'xmlns:w14="http://schemas.microsoft.com/office/word/2010/wordml" '
        'xmlns:wpg="http://schemas.microsoft.com/office/word/2010/wordprocessingGroup" '
        'xmlns:wpi="http://schemas.microsoft.com/office/word/2010/wordprocessingInk" '
        'xmlns:wne="http://schemas.microsoft.com/office/2006/wordml" '
        'xmlns:wps="http://schemas.microsoft.com/office/word/2010/wordprocessingShape" '
        'mc:Ignorable="w14 wp14">'
        "<w:body>"
        + "".join(blocks)
        + sect
        + "</w:body></w:document>"
    )

    document_rels = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/settings" Target="settings.xml"/>
  <Relationship Id="rId3" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_1_overall_comparison.png"/>
  <Relationship Id="rId4" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_2_convergence.png"/>
  <Relationship Id="rId5" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_3_strategy_timeline.png"/>
  <Relationship Id="rId6" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_4_client_trends.png"/>
  <Relationship Id="rId7" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_5_ablation.png"/>
  <Relationship Id="rId8" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_6_multiseed.png"/>
  <Relationship Id="rId9" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="media/fig5_7_bias_correction.png"/>
</Relationships>
"""

    image_names = [
        "fig5_1_overall_comparison.png",
        "fig5_2_convergence.png",
        "fig5_3_strategy_timeline.png",
        "fig5_4_client_trends.png",
        "fig5_5_ablation.png",
        "fig5_6_multiseed.png",
        "fig5_7_bias_correction.png",
    ]

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", package_rels_xml())
        zf.writestr("docProps/core.xml", core_xml())
        zf.writestr("docProps/app.xml", app_xml())
        zf.writestr("word/document.xml", document_xml)
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/settings.xml", settings_xml())
        zf.writestr("word/_rels/document.xml.rels", document_rels)
        for name in image_names:
            zf.write(Path.cwd() / "ch5_figures" / name, f"word/media/{name}")


if __name__ == "__main__":
    output = Path.cwd() / "第五章_结果与讨论_学术版.docx"
    build_docx(output)
    print(output)
