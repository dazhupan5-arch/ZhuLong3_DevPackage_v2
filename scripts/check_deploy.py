import sys; sys.path.insert(0, r'D:\trae_projects\ZhuLong3_DevPackage_v2')
import json; from pathlib import Path

cfg = json.loads(Path(r'D:\trae_projects\ZhuLong3_DevPackage_v2\config\config_agent.json').read_text(encoding='utf-8'))
kn2_cfg = cfg.get('kn2', {})
print(f'kn2.enabled={kn2_cfg.get("enabled")}')
print(f'kn2.shadow_mode={kn2_cfg.get("shadow_mode")}')
print(f'kn2.model_path={kn2_cfg.get("model_path")}')

print()
from zhulong.agent.knowledge_net_kn2 import KN2Inference
kn2 = KN2Inference(r'D:\trae_projects\ZhuLong3_DevPackage_v2\models\kn2_trader.pth')
print(f'KN2: ready={kn2.is_ready} hidden={kn2.hidden_dim} layers={kn2.num_layers}')

print()
from zhulong.agent.trading_agent import TradingAgent
print('TradingAgent import OK')
print('DEPLOYMENT READY')
