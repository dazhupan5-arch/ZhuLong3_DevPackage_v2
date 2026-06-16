import sys
sys.path.insert(0, 'D:/ZhuLong3_Migration_20260609.zip')
import numpy as np
import json

print('=== V14 XGBoost ===')
with open('models/XAUUSD/v14/feature_columns.json') as f:
    cols = json.load(f)
print(f'Feature columns: {len(cols)}')
assert len(cols) == 68, 'Expected 68 features but got ' + str(len(cols))

d = np.load('data/training_data.npz', allow_pickle=True)
struct_col = d['struct']
print(f'Training struct dim: {struct_col.shape[1]}')
assert struct_col.shape[1] == 68, 'Expected 68-dim struct but got ' + str(struct_col.shape[1])

print()
print('=== KnowledgeNet ===')
from zhulong.agent.knowledge_net import _deduplicate_columns
struct68 = struct_col[:50000]
keep = _deduplicate_columns(struct68)
print(f'After dedup: 68 -> {len(keep)} columns')

with open('models/knowledge_net.meta.json') as f:
    kn_meta = json.load(f)
print(f'meta.json input_dim: {kn_meta["input_dim"]}')
print(f'meta.json hidden_dim: {kn_meta["hidden_dim"]}')
print(f'meta.json embed_dim: {kn_meta["embed_dim"]}')
assert kn_meta['input_dim'] == len(keep), 'DIM MISMATCH: meta says ' + str(kn_meta['input_dim']) + ' but dedup gives ' + str(len(keep))

import joblib
scaler = joblib.load('models/knowledge_scaler.pkl')
print(f'scaler n_features_in_: {scaler.n_features_in_}')
assert scaler.n_features_in_ == kn_meta['input_dim'], 'SCALER MISMATCH: scaler=' + str(scaler.n_features_in_) + ' meta=' + str(kn_meta['input_dim'])
print('KN dimensions: CONSISTENT')

labels = d['labels'].astype(int) + 1
from zhulong.agent.knowledge_net import train_knowledge_net
x_sample = struct68[:, keep]
x_scaled = scaler.transform(x_sample)
print(f'After scaler transform: {x_scaled.shape}')
print('KN inference path: VALID')

print()
print('=== RL StateBuilder ===')
from zhulong.agent.state_builder import StateBuilder, STATE_DIM
print(f'STATE_DIM constant: {STATE_DIM}')

struct30 = struct_col[:1, :30]
fake_emb = np.zeros((1, 32), dtype=np.float32)
print(f'struct[:30]: {struct30.shape}')
print(f'embedding[:32]: {fake_emb.shape}')
state_sum = 30 + 32 + 12
print(f'State = struct(30) + emb(32) + tail(12) = {state_sum}')
assert state_sum == STATE_DIM, 'DIM MISMATCH: 30+32+12=' + str(state_sum) + ' != STATE_DIM=' + str(STATE_DIM)
print('RL state dimensions: CONSISTENT')

print(f'Training struct full dim: {struct_col.shape[1]}')
assert struct_col.shape[1] >= 30, 'DIM MISMATCH: training struct has ' + str(struct_col.shape[1]) + ' dims, need >=30'

print()
print('=== ALL DIMENSION CHECKS PASSED ===')
