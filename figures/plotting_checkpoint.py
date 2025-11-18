import pandas as pd
import matplotlib.pyplot as plt
import os, json

# --- CONFIG ---
file_20 = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/checkpoint_responses_2e-6/20_data_res_slow_prms/ft_response_graded.json"
file_50 = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/responses/checkpoint_responses_2e-6/50_data_res_slow_prms/ft_graded.json"
fig_dir = "/n/netscratch/dam_lab/Lab/hdiaz/ft_project/figures/figures_ft/multi_2e-6"
os.makedirs(fig_dir, exist_ok=True)

def load_json_any(path):
    with open(path, "r") as f:
        txt = f.read().strip()
    if "\n" in txt and txt.lstrip().startswith("{"):
        recs = [json.loads(line) for line in txt.splitlines() if line.strip()]
    else:
        recs = json.loads(txt)
    return pd.DataFrame(recs)

def coerce(df):
    for col in ["correctness","clarity","reasoning","notation","total"]:
        if col in df: df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["api_equiv","is_equiv"]:
        if col in df: df[col] = df[col].astype(bool)
    return df

df20 = coerce(load_json_any(file_20)).assign(checkpoint="20%")
df50 = coerce(load_json_any(file_50)).assign(checkpoint="50%")

# --- Headline accuracy metrics ---
acc_rows = []
for label, df in [("20%", df20), ("50%", df50)]:
    acc_rows.append([label, "api_equiv", df["api_equiv"].mean()])
    acc_rows.append([label, "is_equiv", df["is_equiv"].mean()])
    acc_rows.append([label, "Rubric Correct (score=5)", (df["correctness"]==5).mean()])
acc_df = pd.DataFrame(acc_rows, columns=["checkpoint","metric","value"])

plt.figure(figsize=(8,5))
metrics = acc_df["metric"].unique()
x = range(len(metrics))
vals20 = [acc_df[(acc_df.checkpoint=="20%")&(acc_df.metric==m)]["value"].values[0] for m in metrics]
vals50 = [acc_df[(acc_df.checkpoint=="50%")&(acc_df.metric==m)]["value"].values[0] for m in metrics]
w = 0.35
plt.bar([i-w/2 for i in x], vals20, width=w, label="20%")
plt.bar([i+w/2 for i in x], vals50, width=w, label="50%")
plt.xticks(x, metrics, rotation=20, ha="right")
plt.ylim(0,1)
plt.ylabel("Proportion Correct")
plt.title("Accuracy by Method: 20% vs 50%")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "accuracy_compare.png"))
plt.close()

# --- Average rubric scores ---
score_cols = ["correctness","clarity","reasoning","notation","total"]
avg_scores = pd.concat([
    df20[score_cols].mean().to_frame(name="20%"),
    df50[score_cols].mean().to_frame(name="50%")
], axis=1)

avg_scores.plot(kind="bar", figsize=(8,5))
plt.title("Average Rubric Scores: 20% vs 50%")
plt.ylabel("Average Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "avg_scores_compare.png"))
plt.close()

# --- Breakdown by problem type ---
if "type" in df20 and "type" in df50:
    type_compare = pd.concat([
        df20.groupby("type")["total"].mean().to_frame(name="20%"),
        df50.groupby("type")["total"].mean().to_frame(name="50%")
    ], axis=1).fillna(0)

    type_compare.plot(kind="bar", figsize=(10,5))
    plt.title("Average Total Score by Problem Type: 20% vs 50%")
    plt.ylabel("Average Total Score")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "type_compare.png"))
    plt.close()

print("Comparison plots saved in", fig_dir)
