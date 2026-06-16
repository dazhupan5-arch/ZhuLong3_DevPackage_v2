import sys, json; sys.path.insert(0, r'D:\trae_projects\ZhuLong3_DevPackage_v2')
import numpy as np
from zhulong.agent.knowledge_net_kn2 import KN2Inference, encode_position_state

kn2 = KN2Inference(r'D:\trae_projects\ZhuLong3_DevPackage_v2\models\kn2_trader.pth')
print(f'Ready: {kn2.is_ready}')
print(f'hidden={kn2.hidden_dim} layers={kn2.num_layers} embed={kn2.embed_dim}')

raw = np.load(r'D:\trae_projects\ZhuLong3_DevPackage_v2\data\kn2_training_data.npz', allow_pickle=True)
mf = raw['market_feat'].astype(np.float32)

for i in [0, 1000, 5000, 10000, 50000, 100000]:
    dec = kn2.predict(mf[i], encode_position_state())
    nm = dec["action_name"]
    print(f'bar {i:6d}: action={nm:>9s} conf={dec["confidence"]:.3f} trade={dec["should_trade"]} sl={dec["sl_atr_mult"]:.2f} tp={dec["tp_atr_mult"]:.2f}')
print('OK!')
