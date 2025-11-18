"""
Quick analysis of checkpoint performance
"""

# Data from 20_output.out (20 samples)
results_20 = {
    1500: 70.0,
    3000: 75.0,
    4500: 70.0,
    6000: 80.0,  # BEST
    7500: 75.0,
    9000: 55.0,
    10500: 55.0,
    12000: 65.0,
    13500: 45.0,  # WORST
}

# Data from 50_output.out (50 samples - more reliable)
results_50 = {
    1500: 64.0,
    3000: 60.0,
    4500: 58.0,
    6000: 68.0,  # BEST
    7500: 62.0,
    9000: 48.0,
    10500: 48.0,
    12000: 56.0,
    13500: 46.0,  # WORST
}

print("=" * 60)
print("CHECKPOINT PERFORMANCE ANALYSIS")
print("=" * 60)
print("\n📊 Results on 50 samples (more reliable):\n")
print("Step      Accuracy   Change from Previous")
print("-" * 50)

prev = None
for step in sorted(results_50.keys()):
    acc = results_50[step]
    if prev is None:
        change = ""
    else:
        diff = acc - prev
        arrow = "📈" if diff > 0 else "📉" if diff < 0 else "→"
        change = f"{arrow} {diff:+.1f}%"
    
    marker = "  ⭐ BEST" if acc == max(results_50.values()) else ""
    marker = "  ⚠️  WORST" if acc == min(results_50.values()) else marker
    
    print(f"{step:5d}     {acc:5.1f}%    {change:15s}{marker}")
    prev = acc

print("\n" + "=" * 60)
print("KEY FINDINGS:")
print("=" * 60)

best_step = max(results_50.keys(), key=lambda k: results_50[k])
worst_step = min(results_50.keys(), key=lambda k: results_50[k])

print(f"\n✅ BEST checkpoint:  step {best_step} ({results_50[best_step]:.1f}%)")
print(f"❌ WORST checkpoint: step {worst_step} ({results_50[worst_step]:.1f}%)")
print(f"\n📈 Performance gain: {results_50[best_step] - results_50[1500]:.1f}% (from early to peak)")
print(f"📉 Performance loss: {results_50[best_step] - results_50[13500]:.1f}% (from peak to final)")

print("\n" + "=" * 60)
print("INTERPRETATION:")
print("=" * 60)
print("""
1. Training HELPS initially (steps 1500-6000)
   - Peak at step 6000: 68.0%
   
2. Training HURTS after step 6000
   - Steady degradation to 46.0% by step 13500
   - This suggests OVERFITTING to the math training data
   
3. Recommendation:
   - Use checkpoint-6000 for best MATH performance
   - Later checkpoints may have overfit to open-web-math
   - Consider early stopping around step 6000-7500
""")

print("\nNEED BASELINE: What's the pretrained model accuracy?")
print("Run: python simple_eval.py 'microsoft/Phi-4-mini-instruct' --n 50")
print("=" * 60)


