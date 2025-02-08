import matplotlib.pyplot as plt

'''
繪製不同置信度下的各種指標折線圖

'''

# Confidence thresholds and data
confidence_thresholds = [
    0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 
    0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
]
total_correct_predictions = [
    2511, 2511, 2493, 2477, 2456, 2446, 2433, 2418, 2399, 2376, 2344, 
    2300, 2267, 2232, 2198, 2150, 2094, 2036, 1876, 1486
]
total_bird_as_background = [
    354, 354, 372, 388, 409, 419, 432, 447, 466, 489, 521, 
    565, 598, 633, 667, 715, 771, 829, 989, 1379
]
total_background_as_bird = [
    3904, 3904, 2670, 1861, 1381, 1078, 871, 729, 598, 517, 431, 
    374, 307, 261, 226, 180, 144, 114, 69, 30
]

def plot_metrics():
    plt.figure(figsize=(12, 6))

    # Plot total correct predictions
    plt.plot(confidence_thresholds, total_correct_predictions, label="Total Correct Predictions", marker='o')

    # Plot total bird as background
    plt.plot(confidence_thresholds, total_bird_as_background, label="Bird as Background", marker='s')

    # Plot total background as bird
    plt.plot(confidence_thresholds, total_background_as_bird, label="Background as Bird", marker='^')

    # Chart customization
    plt.title("Metrics vs Confidence Threshold")
    plt.xlabel("Confidence Threshold")
    plt.ylabel("Count")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()

    # Save and show the plot
    plt.savefig("metrics_vs_threshold.png")
    plt.show()

if __name__ == "__main__":
    plot_metrics()
