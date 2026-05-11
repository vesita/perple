import tomllib, glob, sys

for task in ['cluster', 'ground', 'wall']:
    print(f'\n=== {task} ===')
    for f in sorted(glob.glob(f'config/bench/{task}/*.toml')):
        try:
            data = tomllib.load(open(f, 'rb'))
            n_full = len(data.get('full', {}).get('params', []))
            n_quick = len(data.get('quick', {}).get('params', []))
            fname = f.replace('\\', '/').split('/')[-1]
            print(f'  {fname:25s} quick={n_quick} full={n_full}')
        except Exception as e:
            print(f'  ERROR: {e}')
