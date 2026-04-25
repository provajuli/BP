import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

df = pd.read_csv('data/data_processing_input/filtered_results.csv')

df['arith_mean'] = (df['sizeA'] + df['sizeC']) / 2
df['geom_mean'] = np.sqrt(df['sizeA'] * df['sizeC'])
df['diff_arith'] = df['sizeB'] - df['arith_mean']
df['diff_geom']  = df['sizeB'] - df['geom_mean']

glyph_types = df['glyph_type'].value_counts().index.tolist()
n = len(glyph_types)
ncols = 3
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols, figsize=(15, nrows * 4))
axes = axes.flatten()

for i, glyph in enumerate(glyph_types):
    ax = axes[i]
    subset = df[df['glyph_type'] == glyph]

    diff = subset['diff_arith']
    mean_diff = diff.mean()
    std_diff  = diff.std()
    count     = len(diff)

    ax.hist(diff, bins=20, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(0,         color='black',  linestyle='--', linewidth=1.2,
               label='Aritmetický střed')
    ax.axvline(mean_diff, color='tomato',  linestyle='-',  linewidth=1.8,
               label=f'Průměr odchylky: {mean_diff:.2f}')

    ax.set_title(f'{glyph}\n(n={count}, σ={std_diff:.2f})', fontsize=11)
    ax.set_xlabel('Odchylka od aritmetického středu', fontsize=9)
    ax.set_ylabel('Počet odpovědí', fontsize=9)
    ax.legend(fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

# Skryj prázdné subploty
for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Odchylky odpovědí od aritmetického středu podle typu glyphu',
             fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('glyph_deviations.png',
            bbox_inches='tight', dpi=150)
plt.savefig('glyph_deviations.pdf',
            bbox_inches='tight', dpi=150)
print("Hotovo")

# Souhrnná tabulka
summary = df.groupby('glyph_type').agg(
    n=('sizeB', 'count'),
    prumer_odchylky_arith=('diff_arith', 'mean'),
    std_arith=('diff_arith', 'std'),
    prumer_odchylky_geom=('diff_geom', 'mean'),
    std_geom=('diff_geom', 'std'),
).round(3)
print("\nSouhrnná tabulka:")
print(summary.to_string())