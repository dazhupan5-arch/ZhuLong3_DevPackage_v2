import sys, json, numpy as np
sys.path.insert(0, 'D:/ZhuLong3_Migration_20260609.zip')

from zhulong.agent.state_builder import StateBuilder

# Load the training data
d = np.load('D:/ZhuLong3_Migration_20260609.zip/data/training_data.npz', allow_pickle=True)
struct = d['struct']
print(f'struct shape: {struct.shape}')

# Generate dummy embeddings (matching 32-dim)
n = len(struct)
emb = np.zeros((n, 32), dtype=np.float32)
emb[:, :min(struct.shape[1], 32)] = struct[:, :32]  # crude approximation

# Build correct 74-dim state: struct[:30] + emb[:32] + account(12)
raw = np.concatenate([struct[:, :30], emb[:, :32], np.zeros((n, 12), dtype=np.float32)], axis=1)
print(f'Combined raw shape: {raw.shape}')  # Should be (n, 74)

# Save scaler
scaler_path = 'D:/ZhuLong3_Migration_20260609.zip/data/agent_state_scaler_xauusd.json'
sb = StateBuilder()
sb.save_scaler(raw[:5000], scaler_path)
print(f'Scaler saved: {scaler_path}')
print(f'Scaler dims: mean={len(sb.mean)}, std={len(sb.std)}')
print('DONE')
