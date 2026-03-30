"""
明代會試《春秋》義搭股分析
============================
本腳本包含三項統計分析：
  1. 獨立樣本t檢定與效果量（Cohen's d, η²）
  2. 變點檢測（PELT, BinSeg, BottomUp, DynProg, CUSUM）
  3. BIC模型選擇

使用方式：
  pip install numpy pandas ruptures statsmodels matplotlib
  python analysis.py

作者：王亭林
日期：2026年3月
"""

import numpy as np
import pandas as pd
import ruptures as rpt
import statsmodels.api as sm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from scipy import stats
import os
import json

# ============================================================
# 0. 設定
# ============================================================
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字型（可依系統環境調整）
FONT_PATHS = [
    '/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',
    '/System/Library/Fonts/PingFang.ttc',
    'C:/Windows/Fonts/msjh.ttc',
]
zh_font = None
for fp in FONT_PATHS:
    if os.path.exists(fp):
        zh_font = FontProperties(fname=fp, size=10)
        break

# ============================================================
# 1. 讀取資料
# ============================================================
def load_data():
    """讀取會試搭股資料，排除題目不全之科次。"""
    df = pd.read_csv(os.path.join(DATA_DIR, 'huishi_data.csv'))
    # 排除題目不全者（n_questions < 3）
    df_full = df[df['n_questions'] >= 3].copy()
    df_full = df_full.sort_values('western_year').reset_index(drop=True)
    return df_full

# ============================================================
# 2. 獨立樣本t檢定與效果量
# ============================================================
def t_test_and_effect_size(df, breakpoint_year=1478):
    """
    以指定年份為分界，進行Welch's t-test，
    並計算Cohen's d與η²。
    """
    pre = df[df['western_year'] <= breakpoint_year]['avg_stocks']
    post = df[df['western_year'] > breakpoint_year]['avg_stocks']
    
    n1, m1, s1 = len(pre), pre.mean(), pre.std(ddof=1)
    n2, m2, s2 = len(post), post.mean(), post.std(ddof=1)
    
    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(pre, post, equal_var=False)
    
    # Welch-Satterthwaite degrees of freedom
    df_welch = ((s1**2/n1 + s2**2/n2)**2 /
                ((s1**2/n1)**2/(n1-1) + (s2**2/n2)**2/(n2-1)))
    
    # Cohen's d（合併標準差）
    sp = np.sqrt(((n1-1)*s1**2 + (n2-1)*s2**2) / (n1+n2-2))
    cohens_d = (m1 - m2) / sp
    
    # Hedges' g（小樣本校正）
    hedges_g = cohens_d * (1 - 3/(4*(n1+n2)-9))
    
    # η²
    eta_sq = t_stat**2 / (t_stat**2 + df_welch)
    
    results = {
        'breakpoint_year': breakpoint_year,
        'pre': {'n': int(n1), 'mean': round(m1, 3), 'sd': round(s1, 3)},
        'post': {'n': int(n2), 'mean': round(m2, 3), 'sd': round(s2, 3)},
        't_stat': round(t_stat, 3),
        'df_welch': round(df_welch, 2),
        'p_value': f'{p_val:.2e}' if p_val < 0.001 else round(p_val, 4),
        'pooled_sd': round(sp, 4),
        'cohens_d': round(cohens_d, 3),
        'hedges_g': round(hedges_g, 3),
        'eta_squared': round(eta_sq, 3),
    }
    
    return results

# ============================================================
# 3. 變點檢測
# ============================================================
def changepoint_detection(df):
    """
    以搭股率與平均比數為雙訊號，
    分別運用五種方法進行變點檢測。
    """
    years = df['western_year'].values
    names = df['era_name'].values
    avg_stocks = df['avg_stocks'].values
    dati_rate = df['dati_rate'].values
    n = len(df)
    
    results = {}
    
    for var_name, signal_1d in [('avg_stocks', avg_stocks), ('dati_rate', dati_rate)]:
        signal = signal_1d.reshape(-1, 1)
        var_results = {}
        
        # --- PELT ---
        algo = rpt.Pelt(model="l2", min_size=3).fit(signal)
        pen_bic = np.log(n) * signal.var()
        bkpts = [i for i in algo.predict(pen=pen_bic) if i < n]
        var_results['PELT'] = {
            'penalty': round(pen_bic, 4),
            'breakpoints': [{'index': b, 'era': names[b-1], 'year': int(years[b-1])} for b in bkpts]
        }
        
        # --- Binary Segmentation ---
        for k in [1, 2, 3]:
            algo = rpt.Binseg(model="l2", min_size=3).fit(signal)
            bkpts = [i for i in algo.predict(n_bkps=k) if i < n]
            var_results[f'BinSeg_k{k}'] = {
                'breakpoints': [{'index': b, 'era': names[b-1], 'year': int(years[b-1])} for b in bkpts]
            }
        
        # --- Bottom-Up ---
        for k in [1, 2, 3]:
            algo = rpt.BottomUp(model="l2", min_size=3).fit(signal)
            bkpts = [i for i in algo.predict(n_bkps=k) if i < n]
            var_results[f'BottomUp_k{k}'] = {
                'breakpoints': [{'index': b, 'era': names[b-1], 'year': int(years[b-1])} for b in bkpts]
            }
        
        # --- Dynamic Programming ---
        for k in [1, 2, 3]:
            algo = rpt.Dynp(model="l2", min_size=3).fit(signal)
            bkpts = [i for i in algo.predict(n_bkps=k) if i < n]
            var_results[f'DynProg_k{k}'] = {
                'breakpoints': [{'index': b, 'era': names[b-1], 'year': int(years[b-1])} for b in bkpts]
            }
        
        # --- CUSUM ---
        X = sm.add_constant(np.arange(n))
        ols = sm.OLS(signal_1d, X).fit()
        cusum = np.cumsum(signal_1d - signal_1d.mean())
        max_idx = int(np.argmax(np.abs(cusum)))
        var_results['CUSUM'] = {
            'max_index': max_idx,
            'max_value': round(float(np.abs(cusum[max_idx])), 4),
            'era': str(names[max_idx]),
            'year': int(years[max_idx]),
        }
        
        # --- BIC Model Selection ---
        bic_scores = {}
        algo_dp = rpt.Dynp(model="l2", min_size=3).fit(signal)
        for k in range(1, 7):
            result = algo_dp.predict(n_bkps=k)
            bkpts_full = [0] + [i for i in result]
            if bkpts_full[-1] != n:
                bkpts_full.append(n)
            rss = sum(
                np.sum((signal_1d[bkpts_full[j]:bkpts_full[j+1]] - 
                        signal_1d[bkpts_full[j]:bkpts_full[j+1]].mean())**2)
                for j in range(len(bkpts_full)-1)
            )
            bic = n * np.log(rss/n) + (2*k+1) * np.log(n)
            bkpts_clean = [i for i in result if i < n]
            bic_scores[k] = {
                'bic': round(bic, 4),
                'breakpoints': [{'index': b, 'era': names[b-1], 'year': int(years[b-1])} for b in bkpts_clean]
            }
        
        best_k = min(bic_scores, key=lambda k: bic_scores[k]['bic'])
        var_results['BIC'] = {'best_k': best_k, 'scores': bic_scores}
        
        # --- Segment statistics for best BIC model ---
        best_bkpts = [0] + [bp['index'] for bp in bic_scores[best_k]['breakpoints']] + [n]
        segments = []
        for j in range(len(best_bkpts)-1):
            seg = signal_1d[best_bkpts[j]:best_bkpts[j+1]]
            segments.append({
                'start': str(names[best_bkpts[j]]),
                'end': str(names[best_bkpts[j+1]-1]) if best_bkpts[j+1] <= n else str(names[-1]),
                'mean': round(float(seg.mean()), 2),
                'std': round(float(seg.std(ddof=1)), 2) if len(seg) > 1 else 0,
                'n': len(seg),
            })
        var_results['segments'] = segments
        
        results[var_name] = var_results
    
    return results

# ============================================================
# 4. 視覺化
# ============================================================
def create_charts(df, cp_results):
    """產生變點檢測視覺化圖表。"""
    names = df['era_name'].values
    years = df['western_year'].values
    avg_stocks = df['avg_stocks'].values
    dati_rate = df['dati_rate'].values
    n = len(df)
    
    fig, axes = plt.subplots(4, 1, figsize=(18, 20), 
                              gridspec_kw={'height_ratios': [2, 2, 1.5, 1.5]})
    
    fp = zh_font if zh_font else FontProperties(size=9)
    fp_title = FontProperties(fname=fp.get_file(), size=13) if zh_font else FontProperties(size=13)
    
    for ax_idx, (var_name, var_label, values, ax) in enumerate([
        ('dati_rate', 'dati rate', dati_rate, axes[0]),
        ('avg_stocks', 'avg stocks / question', avg_stocks, axes[1]),
    ]):
        r = cp_results[var_name]
        best_k = r['BIC']['best_k']
        segments = r['segments']
        bic_bkpts = [bp['index'] for bp in r['BIC']['scores'][best_k]['breakpoints']]
        
        # Bar chart
        colors = []
        palette = ['#3498DB', '#F39C12', '#9B59B6', '#2ECC71']
        seg_boundaries = [0] + bic_bkpts + [n]
        for i in range(n):
            for s_idx in range(len(seg_boundaries)-1):
                if seg_boundaries[s_idx] <= i < seg_boundaries[s_idx+1]:
                    colors.append(palette[s_idx % len(palette)])
                    break
        
        ax.bar(range(n), values, color=colors, alpha=0.8, edgecolor='white', linewidth=0.5)
        
        # Segment means
        for s_idx in range(len(seg_boundaries)-1):
            start, end = seg_boundaries[s_idx], seg_boundaries[s_idx+1]
            seg_mean = values[start:end].mean()
            ax.hlines(seg_mean, start-0.5, end-0.5, colors=palette[s_idx % len(palette)], 
                     linestyles='-', linewidth=2.5, alpha=0.9)
        
        # Breakpoint lines
        for bp in bic_bkpts:
            ax.axvline(x=bp-0.5, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(bp-0.5, ax.get_ylim()[1]*0.95, f'{names[bp-1]}\n({years[bp-1]})', 
                   ha='center', va='top', fontsize=8, color='#E74C3C', fontproperties=fp)
        
        # Legend for segments
        for s_idx, seg in enumerate(segments):
            ax.text(0.98, 0.95 - s_idx*0.08, 
                   f'Seg {s_idx+1}: M={seg["mean"]:.2f}, SD={seg["std"]:.2f}',
                   transform=ax.transAxes, ha='right', va='top', fontsize=9,
                   color=palette[s_idx % len(palette)],
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        title_text = f'BIC best model (k={best_k}) — {var_label}'
        ax.set_title(title_text, fontsize=12, fontweight='bold')
        ax.set_xticks(range(n))
        ax.set_xticklabels([n[:4] for n in names], rotation=60, ha='right', fontproperties=fp, fontsize=7)
        ax.set_ylabel(var_label, fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
    
    # CUSUM charts
    for ax_idx, (var_name, var_label, values, ax) in enumerate([
        ('dati_rate', 'dati rate', dati_rate, axes[2]),
        ('avg_stocks', 'avg stocks', avg_stocks, axes[3]),
    ]):
        cusum = np.cumsum(values - values.mean())
        max_idx = cp_results[var_name]['CUSUM']['max_index']
        
        ax.fill_between(range(n), cusum, 0, where=cusum>=0, alpha=0.3, color='#E74C3C')
        ax.fill_between(range(n), cusum, 0, where=cusum<0, alpha=0.3, color='#3498DB')
        ax.plot(range(n), cusum, 'o-', color='#2C3E50', linewidth=1.2, markersize=3)
        ax.axvline(x=max_idx, color='#F39C12', linestyle='--', linewidth=2, alpha=0.7)
        ax.axhline(y=0, color='gray', linewidth=0.5)
        ax.text(max_idx, cusum[max_idx], f' {names[max_idx]}', fontproperties=fp, fontsize=8, color='#F39C12')
        
        ax.set_title(f'CUSUM — {var_label}', fontsize=11, fontweight='bold')
        ax.set_xticks(range(0, n, 3))
        ax.set_xticklabels([names[i][:4] for i in range(0, n, 3)], rotation=45, ha='right', fontproperties=fp, fontsize=7)
        ax.set_ylabel('CUSUM', fontsize=10)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout(pad=2.0)
    chart_path = os.path.join(OUTPUT_DIR, 'changepoint_analysis.png')
    plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f'Chart saved: {chart_path}')

# ============================================================
# 5. 主程式
# ============================================================
def main():
    print('=' * 70)
    print('明代會試《春秋》義搭股分析')
    print('=' * 70)
    
    # Load data
    df = load_data()
    print(f'\n載入資料：{len(df)}科（排除題目不全者）')
    print(f'時間範圍：{df.iloc[0]["era_name"]}（{df.iloc[0]["western_year"]}）'
          f'至{df.iloc[-1]["era_name"]}（{df.iloc[-1]["western_year"]}）')
    
    # T-test & effect size
    print('\n' + '=' * 70)
    print('一、獨立樣本t檢定與效果量')
    print('=' * 70)
    
    ttest = t_test_and_effect_size(df, breakpoint_year=1478)
    print(f'\n分界年份：{ttest["breakpoint_year"]}')
    print(f'前期：n={ttest["pre"]["n"]}, M={ttest["pre"]["mean"]}, SD={ttest["pre"]["sd"]}')
    print(f'後期：n={ttest["post"]["n"]}, M={ttest["post"]["mean"]}, SD={ttest["post"]["sd"]}')
    print(f'\nWelch\'s t({ttest["df_welch"]}) = {ttest["t_stat"]}, p = {ttest["p_value"]}')
    print(f'合併標準差 Sp = {ttest["pooled_sd"]}')
    print(f'Cohen\'s d = {ttest["cohens_d"]}')
    print(f'Hedges\' g = {ttest["hedges_g"]}')
    print(f'η² = {ttest["eta_squared"]}')
    print(f'\n判定：d={ttest["cohens_d"]} → {"極大效果" if abs(ttest["cohens_d"])>0.8 else "中效果" if abs(ttest["cohens_d"])>0.5 else "小效果"}')
    print(f'判定：η²={ttest["eta_squared"]} → 解釋{ttest["eta_squared"]*100:.1f}%的變異')
    
    # Changepoint detection
    print('\n' + '=' * 70)
    print('二、變點檢測')
    print('=' * 70)
    
    cp = changepoint_detection(df)
    
    for var_name, var_label in [('avg_stocks', '平均比數'), ('dati_rate', '搭股率')]:
        r = cp[var_name]
        best_k = r['BIC']['best_k']
        print(f'\n--- {var_label} ---')
        print(f'BIC最佳變點數：k={best_k}')
        for k, info in r['BIC']['scores'].items():
            bps = ', '.join(f"{bp['era']}({bp['year']})" for bp in info['breakpoints'])
            marker = ' ★' if k == best_k else ''
            print(f'  k={k}: BIC={info["bic"]:.2f}  {bps}{marker}')
        
        print(f'\n各段統計：')
        for i, seg in enumerate(r['segments']):
            print(f'  段{i+1}: {seg["start"]}～{seg["end"]}  M={seg["mean"]}, SD={seg["std"]}, n={seg["n"]}')
        
        print(f'\nCUSUM最大偏離：{r["CUSUM"]["era"]}（{r["CUSUM"]["year"]}），值={r["CUSUM"]["max_value"]}')
    
    # Charts
    print('\n' + '=' * 70)
    print('三、產生圖表')
    print('=' * 70)
    create_charts(df, cp)
    
    # Save all results to JSON
    all_results = {
        'data_info': {
            'n_exams': len(df),
            'year_range': f'{df.iloc[0]["western_year"]}-{df.iloc[-1]["western_year"]}',
        },
        't_test': ttest,
        'changepoint': {
            var: {
                k: v for k, v in cp[var].items() 
                if k not in ['segments']  # segments may have numpy types
            } for var in cp
        },
    }
    
    json_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    print(f'Results saved: {json_path}')
    
    print('\n' + '=' * 70)
    print('分析完成')
    print('=' * 70)

if __name__ == '__main__':
    main()
