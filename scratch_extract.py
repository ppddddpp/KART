import csv
import collections

def parse_summary():
    with open('workspace/temp/case14_snapshot/results/full/case14_merged_summary.csv') as f:
        reader = csv.reader(f)
        next(reader)
        next(reader)
        results = collections.defaultdict(dict)
        for row in reader:
            dataset, model = row[0], row[1]
            results[dataset][model] = {
                'acc_mean': float(row[4]),
                'acc_std': float(row[5]),
                'auroc_mean': float(row[10]),
                'auroc_std': float(row[11]),
            }
        return results

def parse_raw(dataset):
    times = collections.defaultdict(list)
    mems = collections.defaultdict(list)
    with open(f'workspace/temp/case14_snapshot/results/full/{dataset}/raw.csv') as f:
        reader = csv.reader(f)
        headers = next(reader)
        m_idx = headers.index('Model')
        t_idx = headers.index('Train Time')
        mem_idx = headers.index('Peak VRAM (MB)')
        for row in reader:
            model = row[m_idx]
            times[model].append(float(row[t_idx]))
            mems[model].append(float(row[mem_idx]))
    
    avg_times = {m: sum(v)/len(v) for m,v in times.items()}
    avg_mems = {m: sum(v)/len(v) for m,v in mems.items()}
    return avg_times, avg_mems

summ = parse_summary()
for ds, models in summ.items():
    print(f'=== {ds} ===')
    for m, vals in models.items():
        print(f'{m}: Acc {vals[\"acc_mean\"]*100:.2f} \pm {vals[\"acc_std\"]*100:.2f}  AUROC {vals[\"auroc_mean\"]*100:.2f} \pm {vals[\"auroc_std\"]*100:.2f}')

for ds in ['cifar10', 'pathmnist', 'pneumoniamnist']:
    times, mems = parse_raw(ds)
    print(f'=== {ds} ===')
    for m in times.keys():
        print(f'{m}: Time {times[m]:.1f} Mem {mems[m]:.1f}')
