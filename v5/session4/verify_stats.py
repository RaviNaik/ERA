import json, pathlib
for fname in ["wikipedia_stats.json", "sangraha_stats.json"]:
    path = pathlib.Path("data") / fname
    if not path.exists():
        print(fname + ": NOT FOUND")
        continue
    data = json.loads(path.read_text(encoding="utf-8"))
    print("=== " + fname + " ===")
    print("  Run: " + data["run_name"])
    print("  Initial docs: " + str(data["initial_docs"]))
    for s in data["stages"]:
        print("  Stage " + str(s["stage_id"]) + " " + s["stage_name"] +
              ": " + str(s["input_docs"]) + " -> " + str(s["output_docs"]) +
              " (" + str(s["drop_pct"]) + "% dropped) | survival: " +
              str(s["cumulative_survival_pct"]) + "%")
    print()
