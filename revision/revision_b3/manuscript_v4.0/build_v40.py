"""Assemble sn-article-revised-v4.0.tex from v3.2 and the new section files."""
import re, pathlib
V = pathlib.Path('/home/claude/v40')
L = pathlib.Path('/home/claude/v32/sn-article-revised-v3.2.tex').read_text(encoding='utf-8').split('\n')
def seg(a, b):            # 1-based inclusive line range of v3.2
    return '\n'.join(L[a - 1:b])
def rd(n):
    return (V / n).read_text(encoding='utf-8')
def find(pat, start=1):
    for i in range(start - 1, len(L)):
        if pat in L[i]:
            return i + 1
    raise KeyError(pat)

ABSTRACT = r"""\abstract{Quantitative investment aims to identify an optimal security trading strategy, which highlights the critical role of market information in assessing price trends. However, most existing strategies rely primarily on asset-level trading data and neglect cross-firm dependencies arising from shared ownership, investor-relations disclosures, and correlated trading activity. {\color{blue}To address this gap, we propose MGAT-FinRL, which combines three relation graphs (common top-ten shareholders, co-mentions in investor-relations questions, and Kendall correlations of trading value) with a multi-relational graph-attention Transformer (MGAT) that forecasts the next two days' high, low, and closing prices and is trained to rank the return available under next-day execution. Its out-of-sample forecasts enter the state of five deep reinforcement learning agents (PPO, A2C, SAC, TD3, and DDPG) at every decision. On fifteen-stock universes from the Chinese banking and healthcare sectors, the agents were trained on 2022--2023, selected on 2024, and evaluated once on 2025 under a protocol registered before the 2025 data were collected, with next-day execution, trading costs, five paired seeds, and block-bootstrap inference. In 2025, MGAT forecasts the next day's high and low prices 15--29\% more accurately than previous-day values, and its forecasts of the tradable-horizon return have a significant rank correlation with realised returns in healthcare (0.052, $p=0.012$), but not in banking (0.014). Agents that observe the MGAT forecasts do not attain a significantly higher Sharpe ratio than agents without forecast information (paired differences of 0.04 in healthcare and $-0.04$ in banking), and their allocations remain close to equal weighting and nearly insensitive to the forecasts. A ranking rule extracts the forecast information before costs (Sharpe ratio 1.48 versus 1.25 for equal weighting in healthcare) but not after. These results locate the main constraint of the forecast-to-policy pipeline at the interface between a weak cross-sectional signal and the policy learner.}}
"""
INTRO_RES = (r"In 2025, the MGAT forecasts rank the tradable return significantly in healthcare, whereas the agents that observe them do not attain a significantly higher Sharpe ratio than agents without forecasts; policy diagnostics, cost sensitivity, an encoder ablation, and forecast-ranking rules show where the transfer from forecasts to allocations is constrained.")

out = []
t = find(r'\title[Article Title]')
out.append(seg(1, t - 1))
out.append(r'\title[Article Title] {\textcolor{blue}{MGAT-FinRL: A Multi-relational Graph Attention Transformer-Based Reinforcement Learning Investment Strategy}}')
a = find(r'\abstract{'); k = find(r'\keywords{')
out.append(seg(t + 1, a - 1))
out.append(ABSTRACT)
out.append(r'\keywords{\textcolor{blue}{graph attention network, Transformer, deep reinforcement learning, portfolio allocation, financial relation graphs, pre-registered evaluation}}')
i1 = find(r'\section{Introduction}'); ta = find('To address these issues, we propose'); rw = find(r'\section{Related work}')
intro = seg(k + 1, ta - 1)
intro = intro.replace('MGT applies a related principle to financial relations by fusing shareholding, investor-relations Q\\&A, and trading-activity graphs within a single graph encoder.',
                      'MGAT applies a related principle to financial relations by combining shareholding, investor-relations Q\\&A, and trading-activity graphs within a single graph-attention encoder.')
assert 'MGT applies' not in intro
out.append(intro)
out.append(rd('sec_intro_tail.tex').replace('%%RESULTS_INTRO%%', INTRO_RES))
pos = find('Positioning of this study'); arch = find(r'\section{Architecture of MGT-FinRL}')
rel = seg(rw, pos - 1)
rel = rel.replace('whereas MGT learns cross-stock relational representations for price forecasting and supplies its forecasts to a downstream portfolio policy.',
                  'whereas MGAT learns cross-stock relational representations for price forecasting and supplies its forecasts to a downstream portfolio policy.')
assert 'MGT' not in rel.replace('MGT-FinRL', 'X'), [l for l in rel.split('\n') if 'MGT' in l][:2]
out.append(rel)
out.append(rd('sec_positioning.tex'))
out.append(rd('sec_arch.tex'))
# MGAT forecaster: problem definition (v3.2 MGT section up to the relation-graph subsection), relation graphs
mgt = find(r'\section{MGT}'); hrg = find(r'\subsection{Heterogeneous relation graphs}'); mgnn = find(r'\subsection{Multi-layer graph neural network (MGNN)}')
pdx = seg(mgt, hrg - 1)
pdx = pdx.replace(r'\section{MGT}', r'\section{MGAT forecaster}')
pdx = pdx.replace('These limitations motivate the design of the MGT predictor  within MGT-FinRL, which couples multi-relation graph encoding with \\textcolor{blue}{a Transformer encoder for price forecasting}.',
                  'These limitations motivate the design of the MGAT forecaster within MGAT-FinRL, which couples \\textcolor{blue}{graph attention over multiple relation graphs with a Transformer encoder for price forecasting}.')
pdx = pdx.replace('As shown in the upper part of Figure~\\ref{fig:fig1}, MGT integrates GNN-based relational encoding', 'As shown in the upper part of Figure~\\ref{fig:fig1}, \\textcolor{blue}{MGAT} integrates \\textcolor{blue}{graph-attention} relational encoding')
pdx = pdx.replace('To highlight the generality and improvements of MGT over existing architectures', 'To highlight the generality and improvements of \\textcolor{blue}{MGAT} over existing architectures')
pdx = pdx.replace('the proposed MGT further incorporates', 'the proposed \\textcolor{blue}{MGAT} further incorporates')
pdx = pdx.replace('Moreover, rather than modelling each stock in isolation, MGT explicitly', 'Moreover, rather than modelling each stock in isolation, \\textcolor{blue}{MGAT} explicitly')
pdx = pdx.replace('Accordingly, we extend \\eqref{eq:multigcn} and describe MGT as:', 'Accordingly, we extend \\eqref{eq:multigcn} and describe \\textcolor{blue}{MGAT} as:')
pdx = pdx.replace('In the implementation used in this study, MGT forecasts,', 'In the implementation used in this study, MGAT forecasts,')
left = [l for l in pdx.split('\n') if re.search(r'\bMGT\b|MGNN', l)]
assert not left, left[:3]
out.append(pdx)
out.append(seg(hrg, mgnn - 1))
out.append(rd('sec_mgat.tex'))
trf = find(r'\subsection{Temporal Transformer for forecasting}'); imp = find(r'\paragraph{Implemented configuration}', trf); rl = find(r'\section{Reinforcement learning for investment strategy}')
tr = seg(trf, imp - 1)
tr = tr.replace('The transformer block in MGT is designed to model temporal dependencies in stock prices while being conditioned on the relational signals extracted by MGNN.',
                'The \\textcolor{blue}{Transformer encoder \\citep{Vaswani2017Attention}} block in MGAT is designed to model temporal dependencies in stock prices while being conditioned on the relational signals extracted by \\textcolor{blue}{the graph-attention layers}.')
tr = tr.replace('and the MGNN-enhanced feature tensor be $\\mathbf{X}_{\\mathrm{GNN}}\\in\\mathbb{R}^{T\\times N\\times C}$', 'and the \\textcolor{blue}{graph-attention feature tensor} be $\\mathbf{X}_{\\mathrm{GNN}}=\\mathbf{H}^{\\mathrm{GAT}}\\in\\mathbb{R}^{T\\times N\\times C}$')
tr = tr.replace('the \\textcolor{blue}{price forecasts} of each stock\\textcolor{blue}{; the policy learner does not receive this representation}', 'the \\textcolor{blue}{price forecasts} of each stock\\textcolor{blue}{; the policy learner receives the forecasts, not this representation}')
left = [l for l in tr.split('\n') if re.search(r'\bMGT\b|MGNN', l)]
assert not left, left[:3]
out.append(tr)
out.append(rd('sec_objective.tex'))
out.append(rd('sec_rl.tex'))
out.append(rd('sec_data.tex'))
g1 = find(r'\subsection{Relationship graphs}'); fm = find(r'\subsection{Forecasting module: implementation and evaluation}')
out.append(seg(g1, fm - 1))
_ph = rd('sec_posthoc.tex')
import os as _os
if _os.environ.get('DRAFT'):
    _ph = _ph.replace('%%DEPTH%%', '').replace('%%PPO%%', '')
assert '%%' not in _ph.replace('\\%', ''), 'unfilled post-hoc markers'
out.append(rd('sec_results.tex').replace('%%POSTHOC%%', '' if 'PENDING' in _ph else _ph))
out.append(rd('sec_limits.tex'))
dec = find(r'\section*{Declarations}'); apx = find(r'\appendix', dec)
d = seg(dec, apx - 1)
dav = [l for l in d.split('\n') if l.startswith('{\\color{blue}The source code for graph construction')][0]
d = d.replace(dav, r"""{\color{blue}The source code for graph construction, forecasting, data download and ingestion, the backtesting ledger, RL training, and evaluation, together with the registered protocol and its dated amendments, configuration files, random seeds, and the scripts that generate every table and figure of this article, is available at \url{https://github.com/ren148056842/MGT-FinRL} on the branch \texttt{revision-b3}. The repository also contains the forecasts, training histories, and metadata of all forecasting runs, the 2024 validation and 2025 evaluation trajectories of all RL policies and reference strategies, the ingestion checks of the 2025 data, and the analysis outputs. The market data remain subject to the terms of their providers (Tushare, Eastmoney, and Wind) and can be regenerated with the included download scripts; the trained policy checkpoints are available from the corresponding author on request.\par}""")
out.append(d)
out.append(r'\appendix')
out.append(rd('app_models.tex'))
uni = find(r'\section{Stock universes}'); retro = find(r'\section{Retrospective selection diagnostic}'); met = find(r'\section{Portfolio performance metrics}'); bib = find(r'\bibliography{')
out.append('{\\color{blue}\n' + seg(uni, retro - 1).replace('\\par}', '') + '\\par}')
out.append(seg(met, bib - 1))
out.append(seg(bib, len(L)))
tex = '\n'.join(out)
K0 = 'This window ends before the 2023 validation and 2024 evaluation periods of the RL experiments but overlaps the 2022 test year of the first forecasting fold (Section~\\ref{subsec:splits}).'
assert K0 in tex
tex = tex.replace(K0, 'This window ends before the 2024 selection and 2025 evaluation periods of the RL experiments but overlaps the RL training period and the test years of the first two forecasting folds (Section~\\ref{subsec:splits}).')
tex = tex.replace(r'\input{tab_', r'\input{tables/tab_')
# inline the generated tables (the journal template asks not to use \input)
for m in sorted(set(re.findall(r'\\input\{tables/(tab_[a-z0-9_]+)\}', tex))):
    body = (V / 'tables' / (m + '.tex')).read_text(encoding='utf-8').rstrip('\n')
    tex = tex.replace('\\input{tables/%s}' % m, body)
assert not re.search(r'^[^%]*\\input\{', tex, re.M)
(V / 'sn-article-revised-v4.0.tex').write_text(tex, encoding='utf-8')
print('written', len(tex.split('\n')), 'lines')
