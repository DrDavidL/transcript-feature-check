import numpy as np

def debug_spreadsheet_calculation():
    """
    Debug the exact spreadsheet calculation by reverse-engineering the expected values.
    """
    
    print("=== Debugging Spreadsheet Calculation ===\n")
    
    # Expected unnormalized values from spreadsheet
    expected_unnormalized = np.array([
        0.8501434561906863,
        0.012663440749510965, 
        0.0040650406504065045,
        0.06452714236940935,
        0.0125,
        0.03225806451612903,
        0.0006451612903225806,
        0.08108108108108109
    ])
    
    # LR values
    lr_values = np.array([12, 0.3, 0.2, 0.15, 0.25, 0.3, 0.2, 0.5])
    
    # Reverse-engineer the priors used in spreadsheet
    # If unnormalized = prior × LR, then prior = unnormalized / LR
    reverse_engineered_priors = expected_unnormalized / lr_values
    
    print("Expected unnormalized values from spreadsheet:")
    print(expected_unnormalized)
    print(f"\nLR values:")
    print(lr_values)
    print(f"\nReverse-engineered priors (unnormalized / LR):")
    print(reverse_engineered_priors)
    print(f"Sum of reverse-engineered priors: {np.sum(reverse_engineered_priors):.10f}")
    
    # Check if these match the stated priors in spreadsheet
    stated_priors = np.array([0.321, 0.041, 0.02, 0.315, 0.05, 0.1, 0.003, 0.15])
    print(f"\nStated priors from spreadsheet:")
    print(stated_priors)
    print(f"Sum of stated priors: {np.sum(stated_priors):.10f}")
    
    print(f"\nDifferences (reverse-engineered - stated):")
    differences = reverse_engineered_priors - stated_priors
    print(differences)
    
    # Check if the spreadsheet is using different priors
    print(f"\n=== ANALYSIS ===")
    if np.allclose(reverse_engineered_priors, stated_priors, atol=1e-10):
        print("✅ Spreadsheet priors match the calculation")
    else:
        print("❌ Spreadsheet priors DON'T match the calculation!")
        print("   This suggests there's an error in the spreadsheet")
    
    # Let's check what the correct calculation should be with stated priors
    print(f"\n=== CORRECT CALCULATION WITH STATED PRIORS ===")
    correct_unnormalized = stated_priors * lr_values
    correct_norm_constant = np.sum(correct_unnormalized)
    correct_posterior = correct_unnormalized / correct_norm_constant
    
    print(f"Correct unnormalized: {correct_unnormalized}")
    print(f"Correct normalization constant: {correct_norm_constant:.10f}")
    print(f"Correct posterior: {correct_posterior}")
    print(f"Correct posterior sum: {np.sum(correct_posterior):.10f}")
    
    # Compare with spreadsheet expected
    expected_norm_constant = 1.0583266820085506
    expected_posterior = np.array([
        0.8032902039068289, 0.011965531026277813, 0.0038410074313648093,
        0.06097091140794653, 0.012271270495009652, 0.030481927710843374,
        0.0005682966087303988, 0.07661252660397917
    ])
    
    print(f"\n=== COMPARISON WITH SPREADSHEET ===")
    print(f"Normalization constant difference: {correct_norm_constant - expected_norm_constant:.10f}")
    print(f"Posterior differences: {correct_posterior - expected_posterior}")
    print(f"Max posterior difference: {np.max(np.abs(correct_posterior - expected_posterior)):.10f}")
    
    # The key insight: let's see what priors would give the expected results
    print(f"\n=== WHAT PRIORS WOULD GIVE EXPECTED RESULTS? ===")
    
    # If the expected normalization constant is 1.0583266820085506
    # and the LRs are [12, 0.3, 0.2, 0.15, 0.25, 0.3, 0.2, 0.5]
    # then the priors that would give this result are:
    
    # We know: sum(prior_i * LR_i) = normalization_constant
    # And: posterior_i = (prior_i * LR_i) / normalization_constant
    # So: prior_i = (posterior_i * normalization_constant) / LR_i
    
    implied_priors = (expected_posterior * expected_norm_constant) / lr_values
    print(f"Implied priors from expected results: {implied_priors}")
    print(f"Sum of implied priors: {np.sum(implied_priors):.10f}")
    
    print(f"\nDifference from stated priors: {implied_priors - stated_priors}")
    
    return {
        'stated_priors': stated_priors,
        'reverse_engineered_priors': reverse_engineered_priors,
        'implied_priors': implied_priors,
        'correct_calculation': {
            'unnormalized': correct_unnormalized,
            'normalization': correct_norm_constant,
            'posterior': correct_posterior
        }
    }

if __name__ == "__main__":
    results = debug_spreadsheet_calculation()