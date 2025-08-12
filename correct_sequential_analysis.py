import numpy as np

def analyze_sequential_processing():
    """
    Analyze the sequential processing with the correct prior probabilities
    from the user's image.
    """
    
    print("=== Correct Sequential Analysis ===\n")
    
    # Correct prior probabilities from the user's image
    # These are the INITIAL priors, used only for the first feature
    initial_priors = np.array([0.33, 0.041, 0.02, 0.315, 0.05, 0.1, 0.003, 0.15])
    
    print(f"Initial prior probabilities: {initial_priors}")
    print(f"Sum of initial priors: {np.sum(initial_priors):.6f}")
    
    # Features and their LR values (from the spreadsheet)
    features = [
        "Patient Has: Pain relieved with regurgitation",
        "Patient Has: Current heartburn", 
        "Patient Has: Current reflux",
        "Patient Has: family history of RA",
        "Patient does not have: associated shortness of breath"
    ]
    
    lr_values_sequence = [
        np.array([12, 0.3, 0.2, 0.15, 0.25, 0.3, 0.2, 0.5]),      # Pain relieved with regurgitation
        np.array([2.7, 0.5, 1, 1.2, 0.8, 1.6, 1.6, 0.5]),        # Current heartburn
        np.array([3.5, 0.7, 1.1, 1, 1.2, 1.5, 1.2, 0.7]),        # Current reflux
        np.array([1.1, 1.05, 1.05, 0.6, 1, 1, 3.5, 0.7]),        # family history of RA
        np.array([1.2, 0.8, 1, 1.4, 0.2, 0.65, 0.8, 1.1])        # associated shortness of breath
    ]
    
    # Expected results from spreadsheet for first step
    expected_step1_posterior = np.array([
        0.8032902039068289, 0.011965531026277813, 0.0038410074313648093,
        0.06097091140794653, 0.012271270495009652, 0.030481927710843374,
        0.0005682966087303988, 0.07661252660397917
    ])
    
    # Sequential processing
    current_probs = initial_priors.copy()
    
    print(f"\n=== STEP 1: {features[0]} ===")
    lr_values = lr_values_sequence[0]
    print(f"LR values: {lr_values}")
    
    # Our calculation
    unnormalized = current_probs * lr_values
    norm_constant = np.sum(unnormalized)
    posterior = unnormalized / norm_constant
    
    print(f"Prior probabilities: {current_probs}")
    print(f"Unnormalized posterior: {unnormalized}")
    print(f"Normalization constant: {norm_constant:.10f}")
    print(f"Our posterior: {posterior}")
    print(f"Our posterior sum: {np.sum(posterior):.10f}")
    
    print(f"\nExpected from spreadsheet: {expected_step1_posterior}")
    print(f"Expected sum: {np.sum(expected_step1_posterior):.10f}")
    
    print(f"\nDifferences: {posterior - expected_step1_posterior}")
    print(f"Max difference: {np.max(np.abs(posterior - expected_step1_posterior)):.10f}")
    
    # Check if the issue is with the initial priors
    print(f"\n=== DEBUGGING INITIAL PRIORS ===")
    
    # What priors would give the expected result?
    expected_norm_constant = 1.0583266820085506  # From spreadsheet
    implied_priors = (expected_step1_posterior * expected_norm_constant) / lr_values
    
    print(f"Implied priors from expected results: {implied_priors}")
    print(f"Sum of implied priors: {np.sum(implied_priors):.10f}")
    print(f"Difference from our priors: {implied_priors - initial_priors}")
    
    # Test with the priors that sum to 1.009 (as mentioned earlier)
    priors_1009 = np.array([0.33, 0.041, 0.02, 0.315, 0.05, 0.1, 0.003, 0.15])
    if np.sum(priors_1009) > 1.001:
        print(f"\n=== TESTING WITH PRIORS THAT SUM TO 1.009 ===")
        unnorm_1009 = priors_1009 * lr_values
        norm_1009 = np.sum(unnorm_1009)
        post_1009 = unnorm_1009 / norm_1009
        
        print(f"Priors (sum={np.sum(priors_1009):.6f}): {priors_1009}")
        print(f"Unnormalized: {unnorm_1009}")
        print(f"Normalization: {norm_1009:.10f}")
        print(f"Posterior: {post_1009}")
        print(f"Difference from expected: {post_1009 - expected_step1_posterior}")
        print(f"Max difference: {np.max(np.abs(post_1009 - expected_step1_posterior)):.10f}")
    
    # The key insight: check if there's a rounding or precision issue
    print(f"\n=== PRECISION ANALYSIS ===")
    
    # Try with different precision levels
    for precision in [6, 8, 10, 12]:
        rounded_priors = np.round(initial_priors, precision)
        rounded_unnorm = rounded_priors * lr_values
        rounded_norm = np.sum(rounded_unnorm)
        rounded_post = rounded_unnorm / rounded_norm
        
        diff = np.max(np.abs(rounded_post - expected_step1_posterior))
        print(f"Precision {precision}: max diff = {diff:.2e}")
    
    return {
        'our_calculation': posterior,
        'expected': expected_step1_posterior,
        'max_difference': np.max(np.abs(posterior - expected_step1_posterior)),
        'implied_priors': implied_priors
    }

if __name__ == "__main__":
    results = analyze_sequential_processing()
    
    print(f"\n=== CONCLUSION ===")
    if results['max_difference'] < 1e-6:
        print("✅ Calculations match within numerical precision")
    else:
        print(f"❌ Significant difference: {results['max_difference']:.2e}")
        print("   This suggests either:")
        print("   1. Different initial priors were used")
        print("   2. Different calculation method")
        print("   3. Rounding/precision differences")