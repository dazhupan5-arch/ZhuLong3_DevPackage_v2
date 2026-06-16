import psutil
for p in psutil.process_iter(["name", "pid", "create_time"]):
    if "ZhuLong" in p.info["name"]:
        print(f"ZhuLong running: PID={p.info['pid']} started={p.info['create_time']}")
        break
else:
    print("ZhuLong: NOT RUNNING")
