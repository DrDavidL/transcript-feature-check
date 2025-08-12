import numpy as np

def verify_spreadsheet_row():
    """
    Verify the exact calculations from the spreadsheet row provided by the user.
    
    From the image:
    - Feature: "Patient Has: Pain relieved with regurgitation"
    - Diagnostic Grouping: "A gastroesophageal disorder"
    - Category-Prior Probability (pi): 0.321
    - LR of finding result: 12
    - Un-normalized Post-test prob: 0.8501434456
    - Normalization constant: 1.0583266682
    - Category Posterior Probability (qi): 0.8032902039
    - Pointwise KL Term: 1.063032473
    - Entropy reduction per category: 0.272387714
    """
    
    print("=== Verifying Exact Spreadsheet Row ===\n")
    
    # Values from the spreadsheet image
    prior_prob = 0.321
    lr_value = 12
    expected_unnormalized = 0.8501434456
    expected_norm_constant = 1.0583266682
    expected_posterior = 0.8032902039
    expected_kl_term = 1.063032473
    expected_entropy_reduction = 0.272387714
    
    print(f"Spreadsheet values:")
    print(f"  Prior probability: {prior_prob}")
    print(f"  LR value: {lr_value}")
    print(f"  Expected unnormalized: {expected_unnormalized}")
    print(f"  Expected normalization constant: {expected_norm_constant}")
    print(f"  Expected posterior: {expected_posterior}")
    print(f"  Expected KL term: {expected_kl_term}")
    print(f"  Expected entropy reduction: {expected_entropy_reduction}")
    
    # Our calculation for this single category
    our_unnormalized = prior_prob * lr_value
    print(f"\n=== OUR CALCULATION ===")
    print(f"Our unnormalized (prior × LR): {prior_prob} × {lr_value} = {our_unnormalized}")
    
    # Check if this matches the spreadsheet
    unnorm_diff = our_unnormalized - expected_unnormalized
    print(f"Difference in unnormalized: {unnorm_diff:.10f}")
    
    if abs(unnorm_diff) < 1e-6:
        print("✅ Unnormalized calculation matches!")
    else:
        print("❌ Unnormalized calculation differs")
        print(f"   Expected: {expected_unnormalized}")
        print(f"   Our calc: {our_unnormalized}")
        print(f"   This suggests the spreadsheet is using different values")
    
    # Check the posterior calculation
    our_posterior = our_unnormalized / expected_norm_constant
    post_diff = our_posterior - expected_posterior
    
    print(f"\nPosterior check:")
    print(f"Our posterior (unnorm/norm): {our_unnormalized} / {expected_norm_constant} = {our_posterior}")
    print(f"Expected posterior: {expected_posterior}")
    print(f"Difference: {post_diff:.10f}")
    
    if abs(post_diff) < 1e-6:
        print("✅ Posterior calculation matches!")
    else:
        print("❌ Posterior calculation differs")
    
    # The key insight: what prior would give the expected unnormalized value?
    implied_prior = expected_unnormalized / lr_value
    prior_diff = implied_prior - prior_prob
    
    print(f"\n=== REVERSE ENGINEERING ===")
    print(f"What prior would give expected unnormalized?")
    print(f"Implied prior: {expected_unnormalized} / {lr_value} = {implied_prior}")
    print(f"Stated prior: {prior_prob}")
    print(f"Difference: {prior_diff:.10f}")
    
    if abs(prior_diff) < 1e-6:
        print("✅ The spreadsheet is using the stated prior correctly")
    else:
        print("❌ The spreadsheet is NOT using the stated prior")
        print(f"   It's actually using: {implied_prior}")
    
    # Check if 0.321 × 12 should equal 0.8501434456
    expected_simple = 0.321 * 12
    print(f"\n=== SIMPLE VERIFICATION ===")
    print(f"0.321 × 12 = {expected_simple}")
    print(f"Spreadsheet shows: {expected_unnormalized}")
    print(f"Difference: {expected_simple - expected_unnormalized:.10f}")
    
    if abs(expected_simple - expected_unnormalized) < 1e-6:
        print("✅ Basic multiplication is correct")
    else:
        print("❌ Basic multiplication doesn't match spreadsheet")
        print("   This indicates a fundamental calculation error in the spreadsheet")
    
    return {
        'our_unnormalized': our_unnormalized,
        'expected_unnormalized': expected_unnormalized,
        'unnorm_difference': unnorm_diff,
        'implied_prior': implied_prior,
        'stated_prior': prior_prob,
        'basic_calc_matches': abs(expected_simple - expected_unnormalized) < 1e-6
    }

if __name__ == "__main__":
    results = verify_spreadsheet_row()
    
    print(f"\n=== FINAL VERDICT ===")
    if results['basic_calc_matches']:
        print("✅ The spreadsheet's basic calculation (0.321 × 12) is mathematically correct")
        print("   Any differences must be in the overall normalization or sequential processing")
    else:
        print("❌ The spreadsheet has a fundamental calculation error")
        print(f"   0.321 × 12 should equal {0.321 * 12}, not {results['expected_unnormalized']}")