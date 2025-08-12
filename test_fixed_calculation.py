import numpy as np

def test_fixed_calculation():
    """
    Test the fixed calculation that normalizes priors to sum to 1.0
    """
    
    print("=== Testing Fixed Calculation ===\n")
    
    # Original priors that sum to 1.009 (from user's data)
    original_priors = np.array([0.33, 0.041, 0.02, 0.315, 0.05, 0.1, 0.003, 0.15])
    lr_values = np.array([12, 0.3, 0.2, 0.15, 0.25, 0.3, 0.2, 0.5])
    
    print(f"Original priors: {original_priors}")
    print(f"Sum of original priors: {np.sum(original_priors):.6f}")
    print(f"LR values: {lr_values}")
    
    # Apply the fix: normalize priors to sum to 1.0
    prior_sum = np.sum(original_priors)
    normalized_priors = original_priors / prior_sum
    
    print(f"\nNormalized priors: {normalized_priors}")
    print(f"Sum of normalized priors: {np.sum(normalized_priors):.10f}")
    
    # Calculate with normalized priors
    unnormalized = normalized_priors * lr_values
    norm_constant = np.sum(unnormalized)
    posterior = unnormalized / norm_constant
    
    print(f"\n=== CALCULATION WITH NORMALIZED PRIORS ===")
    print(f"Unnormalized posterior: {unnormalized}")
    print(f"Normalization constant: {norm_constant:.10f}")
    print(f"Final posterior: {posterior}")
    print(f"Posterior sum: {np.sum(posterior):.10f}")
    
    # Expected results from spreadsheet
    expected_posterior = np.array([
        0.8032902039068289, 0.011965531026277813, 0.0038410074313648093,
        0.06097091140794653, 0.012271270495009652, 0.030481927710843374,
        0.0005682966087303988, 0.07661252660397917
    ])
    expected_norm_constant = 1.0583266820085506
    
    print(f"\n=== COMPARISON WITH SPREADSHEET ===")
    print(f"Expected posterior: {expected_posterior}")
    print(f"Expected normalization: {expected_norm_constant:.10f}")
    
    print(f"\nDifferences:")
    post_diff = posterior - expected_posterior
    norm_diff = norm_constant - expected_norm_constant
    
    print(f"Posterior differences: {post_diff}")
    print(f"Normalization difference: {norm_diff:.10f}")
    print(f"Max posterior difference: {np.max(np.abs(post_diff)):.10f}")
    
    # Check if this matches better
    if np.max(np.abs(post_diff)) < 1e-6:
        print("\n✅ FIXED! Calculations now match within numerical precision")
    else:
        print(f"\n❌ Still different. Max difference: {np.max(np.abs(post_diff)):.2e}")
        
        # Let's see what the exact issue might be
        print(f"\n=== FURTHER ANALYSIS ===")
        
        # Maybe the spreadsheet is using different LR values or a different calculation method
        # Let's reverse-engineer what would give the expected result
        
        # If posterior = (prior × LR) / norm_constant, then:
        # prior × LR = posterior × norm_constant
        implied_unnormalized = expected_posterior * expected_norm_constant
        implied_priors = implied_unnormalized / lr_values
        
        print(f"Implied unnormalized from expected: {implied_unnormalized}")
        print(f"Implied priors from expected: {implied_priors}")
        print(f"Sum of implied priors: {np.sum(implied_priors):.10f}")
        
        # Check if the spreadsheet might be using unnormalized priors
        print(f"\n=== TESTING WITH UNNORMALIZED PRIORS ===")
        unnorm_unnormalized = original_priors * lr_values
        unnorm_norm_constant = np.sum(unnorm_unnormalized)
        unnorm_posterior = unnorm_unnormalized / unnorm_norm_constant
        
        print(f"With unnormalized priors:")
        print(f"Unnormalized: {unnorm_unnormalized}")
        print(f"Normalization: {unnorm_norm_constant:.10f}")
        print(f"Posterior: {unnorm_posterior}")
        print(f"Difference from expected: {unnorm_posterior - expected_posterior}")
        print(f"Max difference: {np.max(np.abs(unnorm_posterior - expected_posterior)):.10f}")

if __name__ == "__main__":
    test_fixed_calculation()