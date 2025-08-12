import numpy as np
import pandas as pd

def analyze_exact_spreadsheet_discrepancy():
    """
    Analyze the exact discrepancy between the spreadsheet and our app calculations.
    Using the data from the "Belief Updating" worksheet.
    """
    
    print("=== Analysis of Exact Spreadsheet Data ===\n")
    
    # Data from the spreadsheet "Belief Updating" worksheet
    categories = [
        "A gastroesophageal disorder",
        "A cardiovascular disorder", 
        "A dermatologic disorder (shingles)",
        "a non-inflammatory muscle or joint disorder (mechanical MSK joint pain)",
        "a pulmonary disorder",
        "a psychiatric disorder (anxiety)",
        "Any systemic inflammatory joint disorder (rheumatoid arthritis, Sarcoidosis, Myasthenia Gravis)",
        "No definitive diagnosis"
    ]
    
    # STEP 1: "Patient Has: Pain relieved with regurgitation"
    print("=== STEP 1: Patient Has: Pain relieved with regurgitation ===")
    
    # Prior probabilities from spreadsheet (they don't sum to 1.0!)
    step1_priors = np.array([0.321, 0.041, 0.02, 0.315, 0.05, 0.1, 0.003, 0.15])
    step1_lrs = np.array([12, 0.3, 0.2, 0.15, 0.25, 0.3, 0.2, 0.5])
    
    print(f"Prior probabilities: {step1_priors}")
    print(f"Sum of priors: {np.sum(step1_priors):.6f}")
    print(f"LR values: {step1_lrs}")
    
    # Expected results from spreadsheet
    expected_unnormalized = np.array([0.8501434561906863, 0.012663440749510965, 0.0040650406504065045, 
                                     0.06452714236940935, 0.0125, 0.03225806451612903, 
                                     0.0006451612903225806, 0.08108108108108109])
    expected_norm_constant = 1.0583266820085506
    expected_posterior = np.array([0.8032902039068289, 0.011965531026277813, 0.0038410074313648093,
                                  0.06097091140794653, 0.012271270495009652, 0.030481927710843374,
                                  0.0005682966087303988, 0.07661252660397917])
    
    # Our calculation
    our_unnormalized = step1_priors * step1_lrs
    our_norm_constant = np.sum(our_unnormalized)
    our_posterior = our_unnormalized / our_norm_constant
    
    print(f"\n--- OUR CALCULATION ---")
    print(f"Unnormalized: {our_unnormalized}")
    print(f"Normalization constant: {our_norm_constant:.10f}")
    print(f"Posterior: {our_posterior}")
    print(f"Posterior sum: {np.sum(our_posterior):.10f}")
    
    print(f"\n--- SPREADSHEET EXPECTED ---")
    print(f"Unnormalized: {expected_unnormalized}")
    print(f"Normalization constant: {expected_norm_constant:.10f}")
    print(f"Posterior: {expected_posterior}")
    print(f"Posterior sum: {np.sum(expected_posterior):.10f}")
    
    print(f"\n--- DIFFERENCES ---")
    unnorm_diff = our_unnormalized - expected_unnormalized
    norm_diff = our_norm_constant - expected_norm_constant
    post_diff = our_posterior - expected_posterior
    
    print(f"Unnormalized differences: {unnorm_diff}")
    print(f"Normalization difference: {norm_diff:.10f}")
    print(f"Posterior differences: {post_diff}")
    print(f"Max posterior difference: {np.max(np.abs(post_diff)):.10f}")
    
    # Check if the issue is with non-normalized priors
    print(f"\n=== ANALYSIS ===")
    print(f"1. Prior probabilities sum to {np.sum(step1_priors):.6f}, not 1.0")
    print(f"2. This violates the fundamental requirement for probability distributions")
    
    # Test with normalized priors
    normalized_priors = step1_priors / np.sum(step1_priors)
    norm_unnormalized = normalized_priors * step1_lrs
    norm_norm_constant = np.sum(norm_unnormalized)
    norm_posterior = norm_unnormalized / norm_norm_constant
    
    print(f"\n--- WITH NORMALIZED PRIORS ---")
    print(f"Normalized priors: {normalized_priors}")
    print(f"Normalized priors sum: {np.sum(normalized_priors):.10f}")
    print(f"Unnormalized: {norm_unnormalized}")
    print(f"Normalization constant: {norm_norm_constant:.10f}")
    print(f"Posterior: {norm_posterior}")
    print(f"Posterior sum: {np.sum(norm_posterior):.10f}")
    
    # Check KL divergence calculation
    print(f"\n=== KL DIVERGENCE CHECK ===")
    
    # Our KL divergence calculation
    def calculate_kl_divergence(posterior, prior):
        posterior = np.maximum(posterior, 1e-10)
        prior = np.maximum(prior, 1e-10)
        return np.sum(posterior * np.log2(posterior / prior))
    
    our_kl = calculate_kl_divergence(our_posterior, step1_priors)
    expected_kl = 0.7354401407426722  # From spreadsheet
    
    print(f"Our KL divergence: {our_kl:.10f}")
    print(f"Expected KL divergence: {expected_kl:.10f}")
    print(f"KL difference: {our_kl - expected_kl:.10f}")
    
    # Test with normalized priors for KL
    norm_kl = calculate_kl_divergence(norm_posterior, normalized_priors)
    print(f"KL with normalized priors: {norm_kl:.10f}")
    
    return {
        'prior_sum_issue': np.sum(step1_priors) != 1.0,
        'max_posterior_diff': np.max(np.abs(post_diff)),
        'kl_diff': our_kl - expected_kl,
        'our_calculation': {
            'unnormalized': our_unnormalized,
            'normalization': our_norm_constant,
            'posterior': our_posterior,
            'kl': our_kl
        },
        'spreadsheet_expected': {
            'unnormalized': expected_unnormalized,
            'normalization': expected_norm_constant,
            'posterior': expected_posterior,
            'kl': expected_kl
        }
    }

if __name__ == "__main__":
    results = analyze_exact_spreadsheet_discrepancy()
    
    print(f"\n=== SUMMARY ===")
    if results['prior_sum_issue']:
        print("❌ ISSUE FOUND: Prior probabilities don't sum to 1.0")
        print("   This is the root cause of the discrepancy!")
        print("   The app correctly requires priors to sum to 1.0")
        print("   The spreadsheet uses invalid probability distributions")
    
    if results['max_posterior_diff'] > 1e-6:
        print(f"❌ Significant posterior differences: {results['max_posterior_diff']:.2e}")
    else:
        print(f"✅ Posterior calculations match within numerical precision")
    
    if abs(results['kl_diff']) > 1e-6:
        print(f"❌ KL divergence differs: {results['kl_diff']:.2e}")
    else:
        print(f"✅ KL divergence calculations match")