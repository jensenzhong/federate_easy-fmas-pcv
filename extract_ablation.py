import os, re, glob

base = r'c:\Users\23079\Desktop\科研\federate_easy - 212 - 副本\results\ablation_logs'
files = sorted(glob.glob(os.path.join(base, '*.log')))

for f in files:
    name = os.path.basename(f)
    with open(f, encoding='utf-8', errors='ignore') as fh:
        content = fh.read()
    lines = content.splitlines()
    mape = rmse = mae = mpe = r2 = '?'
    for line in reversed(lines):
        if re.search(r'MAPE:\s', line) and 'Best' not in line and 'Val' not in line and mape == '?':
            m = re.search(r'([0-9]+\.[0-9]+)%', line)
            if m: mape = m.group(1)
        if 'RMSE:' in line and 'NRMSE' not in line and rmse == '?':
            m = re.search(r'\$([\d,]+\.\d+)', line)
            if m: rmse = m.group(1).replace(',', '')
        if 'MAE:' in line and mae == '?':
            m = re.search(r'\$([\d,]+\.\d+)', line)
            if m: mae = m.group(1).replace(',', '')
        if 'MPE:' in line and mpe == '?':
            m = re.search(r'([\-0-9]+\.[0-9]+)%', line)
            if m: mpe = m.group(1)
        if re.search(r'R2:\s', line) and r2 == '?':
            m = re.search(r'([\-0-9]+\.[0-9]+)', line)
            if m: r2 = m.group(1)
    print(f'{name}|{mape}|{rmse}|{mae}|{mpe}|{r2}')
