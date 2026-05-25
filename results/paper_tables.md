# Paper Tables (LaTeX Source)

Generated from the current CSV outputs. Main B/C rows use multi-seed summaries when available.

```latex

\begin{table}[htbp]
\centering
\caption{Performance Comparison of Different Scenarios on Highway Cost Prediction}
\label{tab:main_results}
\begin{tabular}{lcccccc}
\toprule
Scenario & N & MAPE (\%) $\downarrow$ & RMSE (\$) $\downarrow$ & MAE (\$) $\downarrow$ & MPE (\%) & $R^2$ $\uparrow$ \\
\midrule
A (GBR) & 1 & 64.33 & 1,639,791 & 1,189,341 & 29.67 & - \\
A' (NN) & 1 & 50.58 & 1,406,208 & 1,018,121 & 19.84 & - \\
B (FedAvg) & 5 & 51.63 $\pm$ 5.73 & 1,716,607 $\pm$ 122,275 & 1,242,667 $\pm$ 131,528 & -12.05 $\pm$ 13.05 & 0.3038 $\pm$ 0.1027 \\
C (MAS-FL-LLM) & 5 & 50.14 $\pm$ 4.66 & 1,699,480 $\pm$ 181,203 & 1,223,061 $\pm$ 162,467 & -11.31 $\pm$ 14.07 & 0.3142 $\pm$ 0.1476 \\
C + Bias Corr. & 5 & 50.20 $\pm$ 4.48 & 1,534,372 $\pm$ 61,640 & 1,091,372 $\pm$ 53,981 & 3.25 $\pm$ 5.56 & 0.4453 $\pm$ 0.0448 \\

\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Ablation Study: Component Contribution Analysis}
\label{tab:ablation}
\begin{tabular}{lccccl}
\toprule
Config & FedProx & Strategy & LLM & MAPE (\%) & Contribution \\
\midrule
B-baseline & No & size\_only & No & 51.63 $\pm$ 5.73 & Baseline \\
B+FedProx & Yes & size\_only & No & 51.62 $\pm$ 5.73 & +FedProx regularization \\
C-fixed-perf & Yes & perf\_only & No & 51.54 $\pm$ 5.96 & +Performance weighting \\
C-fixed-hybrid & Yes & hybrid & No & 51.69 $\pm$ 5.80 & +Hybrid weighting \\

\bottomrule
\end{tabular}
\end{table}

\begin{table}[htbp]
\centering
\caption{Stratified Performance by Project Size}
\label{tab:stratified}
\begin{tabular}{llccccc}
\toprule
Scenario & Size Category & N & MAPE (\%) & RMSE (\$) & MAE (\$) & MPE (\%) \\
\midrule
A (GBR) & Small (<$1M) & 8 & 275.54 & 1,603,961 & 1,079,766 & 269.09 \\
 & Medium ($1M-$5M) & 50 & 38.30 & 1,276,245 & 972,440 & 5.52 \\
 & Large (>=$5M) & 12 & 32.01 & 2,680,211 & 2,166,149 & -29.32 \\
\midrule
A' (NN) & Small (<$1M) & 8 & 183.95 & 1,071,201 & 742,405 & 182.53 \\
 & Medium ($1M-$5M) & 50 & 34.68 & 1,135,942 & 855,467 & 4.49 \\
 & Large (>=$5M) & 12 & 27.91 & 2,322,384 & 1,879,657 & -24.70 \\
\midrule

\bottomrule
\end{tabular}
\end{table}
```
