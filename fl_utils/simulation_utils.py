import json
import os
from pathlib import Path
import pandas as pd

def save_simulation_metrics(stage: str, data: dict, output_dir="fl_results/metrics"):
    """
    Save metrics for a specific simulation stage.
    
    Args:
        stage: 'before_fl', 'fl_round_X', 'after_personalization'
        data: Dictionary of metrics (e.g., {"A": 0.78, "B": 0.92})
        output_dir: Directory to save JSONs
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save raw JSON
    filepath = Path(output_dir) / f"{stage}.json"
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"✓ Metrics saved: {filepath}")

def generate_comparison_table(metrics_dir="fl_results/metrics"):
    """
    Compile all stage metrics into a final comparison table.
    Expects specific JSON filenames.
    """
    metrics_path = Path(metrics_dir)
    
    # Load stages
    try:
        with open(metrics_path / "before_fl.json") as f:
            before = json.load(f)
        with open(metrics_path / "after_fl.json") as f:
            fl = json.load(f)
        with open(metrics_path / "after_personalization.json") as f:
            pers = json.load(f)
            
        # Combine into DataFrame rows
        rows = []
        hospitals = sorted(list(set(before.keys()) | set(fl.keys()) | set(pers.keys())))
        
        for h in hospitals:
            b_val = before.get(h, 0.5)
            f_val = fl.get(h, 0.5)
            p_val = pers.get(h, 0.5)
            
            rows.append({
                "Hospital": h,
                "Local (Baseline)": f"{b_val:.4f}",
                "After FL": f"{f_val:.4f}",
                "After Personalization": f"{p_val:.4f}",
                "Gain (Net)": f"{p_val - b_val:+.4f}"
            })
            
        # df = pd.DataFrame(rows)
        # md_table = df.to_markdown(index=False)
        
        # Manual Markdown Table Generation to avoid tabulate dependency
        headers = ["Hospital", "Local (Baseline)", "After FL", "After Personalization", "Gain (Net)"]
        header_row = "| " + " | ".join(headers) + " |"
        separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
        
        md_rows = []
        for r in rows:
            row_str = f"| {r['Hospital']} | {r['Local (Baseline)']} | {r['After FL']} | {r['After Personalization']} | {r['Gain (Net)']} |"
            md_rows.append(row_str)
            
        md_table = "\n".join([header_row, separator_row] + md_rows)
        
        # Save as Markdown artifact
        report_path = "fl_results/FINAL_SIMULATION_REPORT.md"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# FL Simulation Results: The 3-Stage Boost\n\n")
            f.write(md_table)
            
        # print(f"\n{md_table}\n") # Squelch print to avoid console encoding errors
        print(f"Report generated: {report_path}")
        
    except FileNotFoundError as e:
        print(f"⚠️ Could not generate table: Missing metric files. {e}")

