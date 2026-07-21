import pandas as pd

d = pd.read_csv('sweep_1p4b.csv')
v = d[d.domain == 'val']
s = v[(v.dH_move >= 1) & (v.h_esc >= 2 * v.h_st_esc)].sort_values(
    'h_esc', ascending=False)

cols = ['layer', 'qhead', 'h_esc', 'h_st_esc',
        'cov4_flow', 'cov4_static', 'dH_move', 'hub_mass']

print(f'STRUCTURED HEADS: {len(s)}')
print('\nTOP 5')
print(s.head()[cols].to_string(index=False))
print('\nBOTTOM 5')
print(s.tail()[cols].to_string(index=False))