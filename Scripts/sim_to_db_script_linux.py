import os, glob
import pandas as pd
import numpy as np

try:
    from scapy.all import rdpcap, IP, RadioTap
except ImportError:
    os.system("pip install scapy --quiet")
    from scapy.all import rdpcap, IP, RadioTap

BASELINE_DIR = '/home/tareqd5/IoD_Sim/results/baseline-2026-05-22.16-30-34'
ATTACK_DIR   = '/home/tareqd5/IoD_Sim/results/mitm_attack-2026-05-22.16-30-55'

def extract_packets(results_dir, label):
    records = []
    for pcap_file in sorted(glob.glob(os.path.join(results_dir, '*.pcap'))):
        node_name = os.path.basename(pcap_file).replace('.pcap', '')
        try:
            pkts = rdpcap(pcap_file)
        except Exception as e:
            print(f"  Erreur {pcap_file}: {e}")
            continue
        for p in pkts:
            if not p.haslayer(IP):
                continue
            rssi = None
            if p.haslayer(RadioTap):
                rssi = getattr(p[RadioTap], 'dBm_AntSignal', None)
            records.append({
                'timestamp' : float(p.time),
                'src_ip'    : p[IP].src,
                'dst_ip'    : p[IP].dst,
                'pkt_len'   : len(p),
                'ttl'       : p[IP].ttl,
                'ip_id'     : p[IP].id,
                'proto'     : p[IP].proto,
                'rssi_dbm'  : rssi,
                'node'      : node_name,
                'label'     : label,
            })
    return records

print("Extraction baseline (label=0)...")
r0 = extract_packets(BASELINE_DIR, 0)
print(f"  {len(r0)} paquets")

print("Extraction attaque (label=1)...")
r1 = extract_packets(ATTACK_DIR, 1)
print(f"  {len(r1)} paquets")

df = pd.DataFrame(r0 + r1).sort_values(['label','timestamp']).reset_index(drop=True)
print(f"Total brut : {len(df)} paquets")

WINDOW = 5.0

def windowed_features(df_sub):
    rows = []
    df_sub = df_sub.sort_values('timestamp').reset_index(drop=True)
    t_min, t_max = df_sub['timestamp'].min(), df_sub['timestamp'].max()
    t = t_min
    while t < t_max:
        w = df_sub[(df_sub['timestamp'] >= t) & (df_sub['timestamp'] < t + WINDOW)]
        if len(w) >= 3:
            times = w['timestamp'].values
            ipd   = np.diff(times)
            rssi  = w['rssi_dbm'].dropna().values
            rows.append({
                'window_start'  : round(t, 2),
                'src_ip'        : w['src_ip'].mode()[0],
                'n_packets'     : len(w),
                'pkt_rate'      : len(w) / WINDOW,
                'mean_ipd_ms'   : np.mean(ipd)*1000 if len(ipd) else 0,
                'jitter_ms'     : np.std(ipd)*1000  if len(ipd) else 0,
                'rssi_mean_dbm' : np.mean(rssi) if len(rssi) else np.nan,
                'rssi_std_dbm'  : np.std(rssi)  if len(rssi) else np.nan,
                'mean_pkt_len'  : w['pkt_len'].mean(),
                'std_pkt_len'   : w['pkt_len'].std(),
                'ttl_mean'      : w['ttl'].mean(),
                'ttl_std'       : w['ttl'].std() if len(w)>1 else 0,
                'n_unique_src'  : w['src_ip'].nunique(),
                'n_unique_dst'  : w['dst_ip'].nunique(),
                'dup_ip_id'     : int(w.duplicated(subset=['ip_id','src_ip']).sum()),
                'label'         : int(w['label'].mode()[0]),
                'node'          : w['node'].mode()[0],
            })
        t += WINDOW / 2  # chevauchement 50%
    return rows

print("\nCalcul features (fenêtres 5s, chevauchement 50%)...")
all_rows = []
for (node, label), grp in df.groupby(['node','label']):
    all_rows.extend(windowed_features(grp))

feat_df = pd.DataFrame(all_rows).dropna(subset=['pkt_rate'])
n0 = (feat_df['label']==0).sum()
n1 = (feat_df['label']==1).sum()
print(f"Dataset features : {len(feat_df)} fenêtres")
print(f"  label=0 (normal)  : {n0}")
print(f"  label=1 (attaque) : {n1}")
print(f"  ratio déséquilibre: {max(n0,n1)/max(min(n0,n1),1):.1f}x")

out = '/home/tareqd5/IoD_Sim/dataset_mitm.csv'
feat_df.to_csv(out, index=False)
