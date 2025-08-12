import numpy as np

def analyze_spreadsheet_formula():
    """
    Analyze the spreadsheet formula: =IF(AND(C2>0,C2<1,D2>0), ...)
    
    This suggests the spreadsheet is using conditional logic that only calculates
    the unnormalized posterior if:
    - C2 > 0 (prior probability is positive)
    - C2 < 1 (prior probability is less than 1)
    - D2 > 0 (LR value is positive)
    
    The incomplete formula suggests there might be different calculation logic.
    """
    
    print("=== Analyzing Spreadsheet Formula Logic ===\n")
    
    # Values from the spreadsheet
    prior_prob = 0.321  # C2
    lr_value = 12       # D2
    expected_unnormalized = 0.8501434456
    
    print(f"Spreadsheet values:")
    print(f"  Prior probability (C2): {prior_prob}")
    print(f"  LR value (D2): {lr_value}")
    print(f"  Reported unnormalized: {expected_unnormalized}")
    
    # Check the conditions
    condition1 = prior_prob > 0
    condition2 = prior_prob < 1
    condition3 = lr_value > 0
    
    print(f"\nFormula conditions:")
    print(f"  C2 > 0: {condition1} ({prior_prob} > 0)")
    print(f"  C2 < 1: {condition2} ({prior_prob} < 1)")
    print(f"  D2 > 0: {condition3} ({lr_value} > 0)")
    print(f"  All conditions met: {condition1 and condition2 and condition3}")
    
    if condition1 and condition2 and condition3:
        print("✅ All conditions are satisfied, so the formula should execute")
    else:
        print("❌ Some conditions are not met")
    
    # The key question: what calculation would give 0.8501434456?
    print(f"\n=== REVERSE ENGINEERING THE FORMULA ===")
    
    # Possibility 1: Different prior being used
    implied_prior_simple = expected_unnormalized / lr_value
    print(f"If simple multiplication: {expected_unnormalized} / {lr_value} = {implied_prior_simple}")
    
    # Possibility 2: Using odds instead of probabilities
    # If prior is converted to odds: odds = p / (1-p)
    prior_odds = prior_prob / (1 - prior_prob)
    odds_calculation = prior_odds * lr_value
    print(f"If using odds: {prior_prob}/(1-{prior_prob}) × {lr_value} = {prior_odds:.6f} × {lr_value} = {odds_calculation:.6f}")
    
    # Possibility 3: Converting back from odds to probability
    # If odds_result = prior_odds × LR, then prob = odds_result / (1 + odds_result)
    if odds_calculation > 0:
        prob_from_odds = odds_calculation / (1 + odds_calculation)
        print(f"Converting odds back to probability: {odds_calculation:.6f} / (1 + {odds_calculation:.6f}) = {prob_from_odds:.6f}")
        
        odds_diff = abs(prob_from_odds - expected_unnormalized)
        print(f"Difference from expected: {odds_diff:.10f}")
        
        if odds_diff < 1e-6:
            print("✅ FOUND IT! The spreadsheet is using ODDS FORM of Bayes' theorem")
            print("   Formula: posterior_odds = prior_odds × LR")
            print("   Then: posterior_prob = posterior_odds / (1 + posterior_odds)")
        else:
            print("❌ Odds calculation doesn't match either")
    
    # Possibility 4: Some other transformation
    print(f"\n=== OTHER POSSIBILITIES ===")
    
    # Maybe it's using a different base or transformation
    log_calc = np.log(prior_prob) + np.log(lr_value)
    exp_calc = np.exp(log_calc)
    print(f"Log space calculation: exp(ln({prior_prob}) + ln({lr_value})) = {exp_calc:.6f}")
    
    # Maybe it's normalizing by something else
    normalized_by_lr = (prior_prob * lr_value) / lr_value
    print(f"Normalized by LR: ({prior_prob} × {lr_value}) / {lr_value} = {normalized_by_lr:.6f}")
    
    # Check if it's using a different prior from a different step
    print(f"\n=== SEQUENTIAL PROCESSING CHECK ===")
    print("The spreadsheet might be using the posterior from a previous step as the prior")
    print("Let's check what prior would give the expected result:")
    
    # If this is step N, what was the posterior from step N-1?
    # We know from earlier analysis that the implied prior is ~0.0708
    previous_posterior = 0.07084528713333334
    sequential_calc = previous_posterior * lr_value
    sequential_prob = sequential_calc / (1 + sequential_calc)
    
    print(f"If prior was {previous_posterior:.6f} (from previous step):")
    print(f"  Odds calculation: {previous_posterior:.6f} × {lr_value} = {sequential_calc:.6f}")
    print(f"  Back to probability: {sequential_calc:.6f} / (1 + {sequential_calc:.6f}) = {sequential_prob:.6f}")
    print(f"  Difference from expected: {abs(sequential_prob - expected_unnormalized):.10f}")
    
    return {
        'conditions_met': condition1 and condition2 and condition3,
        'odds_calculation': prob_from_odds if 'prob_from_odds' in locals() else None,
        'odds_matches': odds_diff < 1e-6 if 'odds_diff' in locals() else False,
        'implied_prior': implied_prior_simple
    }

if __name__ == "__main__":
    results = analyze_spreadsheet_formula()
    
    print(f"\n=== CONCLUSION ===")
    if results.get('odds_matches', False):
        print("🎯 SOLUTION FOUND: The spreadsheet is using the ODDS FORM of Bayes' theorem")
        print("   This is a different (but valid) approach to Bayesian updating")
        print("   Formula: posterior_odds = prior_odds × LR")
        print("   Then: posterior_probability = posterior_odds / (1 + posterior_odds)")
        print("\n   This explains the discrepancy - it's a different mathematical approach!")
    else:
        print("❓ The exact formula logic is still unclear")
        print("   More information needed about the complete spreadsheet formula")