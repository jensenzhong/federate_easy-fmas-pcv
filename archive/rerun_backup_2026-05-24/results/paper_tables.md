# Paper Tables (LaTeX Source)

Copy these LaTeX tables into your paper.

```latex

\begin{table}[htbp]
\centering
\caption{Performance Comparison of Different Scenarios on Highway Cost Prediction}
\label{tab:main_results}
\begin{tabular}{lccccc}
\toprule
Scenario & MAPE (\%) $\downarrow$ & RMSE (\$) $\downarrow$ & MAE (\$) $\downarrow$ & MPE (\%) & $R^2$ $\uparrow$ \\
\midrule
A (GBR) & 64.33 & 1,639,791 & 1,189,341 & 29.67 & - \\
A' (NN) & 50.58 & 1,406,208 & 1,018,121 & 19.84 & - \\
B (FedAvg) & 55.42 & 1,555,598 & 1,096,809 & 9.99 & 0.4306 \\
C (MAS-FL-LLM) & 55.29 & 1,925,106 & 1,463,514 & -31.52 & 0.1279 \\
\midrule
C + Bias Corr. & 52.48 & 1,543,439 & 1,121,960 & 0.13 & 0.4394 \\

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
B-baseline & No & size\_only & No & - & Baseline \\
B+FedProx & Yes & size\_only & No & - & +FedProx regularization \\
C-perf\_only & Yes & perf\_only & No & - & +Performance weighting \\
C-hybrid & Yes & hybrid & No & - & +Hybrid weighting \\
C-LLM & Yes & Dynamic & Yes & - & +LLM decision making \\
C-LLM+bias & Yes & Dynamic & Yes & - & +Bias correction \\

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
