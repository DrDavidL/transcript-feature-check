import numpy as np

def compare_approaches_for_multi_category():
    """
    Compare odds form vs probability form for multiple diagnostic categories
    to determine which is more mathematically appropriate.
    """
    
    print("=== Comparing Approaches for Multiple Diagnostic Categories ===\n")
    
    # Sample data: 3 diagnostic categories
    categories = ["Gastroesophageal", "Cardiovascular", "Dermatologic"]
    priors = np.array([0.4, 0.3, 0.3])  # Must sum to 1.0
    lrs = np.array([6.0, 0.3, 0.6])
    
    print(f"Categories: {categories}")
    print(f"Prior probabilities: {priors} (sum = {np.sum(priors):.3f})")
    print(f"Likelihood ratios: {lrs}")
    
    print(f"\n=== APPROACH 1: PROBABILITY FORM (Current App) ===")
    # Standard multi-category Bayesian updating
    unnormalized = priors * lrs
    norm_constant = np.sum(unnormalized)
    prob_posterior = unnormalized / norm_constant
    
    print(f"Unnormalized: {unnormalized}")
    print(f"Normalization constant: {norm_constant:.6f}")
    print(f"Posterior probabilities: {prob_posterior}")
    print(f"Sum: {np.sum(prob_posterior):.6f}")
    
    print(f"\n=== APPROACH 2: ODDS FORM (Spreadsheet Style) ===")
    # Convert each prior to odds, apply LR, convert back
    odds_posteriors = []
    for i in range(len(priors)):
        prior_odds = priors[i] / (1 - priors[i])
        posterior_odds = prior_odds * lrs[i]
        posterior_prob = posterior_odds / (1 + posterior_odds)
        odds_posteriors.append(posterior_prob)
        
        print(f"{categories[i]}:")
        print(f"  Prior odds: {priors[i]:.3f} / (1 - {priors[i]:.3f}) = {prior_odds:.6f}")
        print(f"  Posterior odds: {prior_odds:.6f} × {lrs[i]} = {posterior_odds:.6f}")
        print(f"  Back to probability: {posterior_odds:.6f} / (1 + {posterior_odds:.6f}) = {posterior_prob:.6f}")
    
    odds_posteriors = np.array(odds_posteriors)
    print(f"\nOdds-based posteriors: {odds_posteriors}")
    print(f"Sum: {np.sum(odds_posteriors):.6f}")
    
    print(f"\n=== CRITICAL ISSUE WITH ODDS FORM ===")
    if abs(np.sum(odds_posteriors) - 1.0) > 0.001:
        print(f"❌ PROBLEM: Odds-based posteriors don't sum to 1.0!")
        print(f"   Sum = {np.sum(odds_posteriors):.6f}")
        print(f"   This violates the fundamental requirement that probabilities sum to 1.0")
        print(f"   The results are not valid probabilities for competing diagnoses")
    else:
        print(f"✅ Odds-based posteriors sum to 1.0")
    
    print(f"\n=== MATHEMATICAL ANALYSIS ===")
    print(f"The odds form treats each category independently:")
    print(f"  - Each prior is converted to odds against 'not having this condition'")
    print(f"  - LR is applied to these individual odds")
    print(f"  - Results are converted back to individual probabilities")
    print(f"  - BUT these don't account for the constraint that all probabilities must sum to 1.0")
    
    print(f"\nThe probability form treats categories as competing:")
    print(f"  - All categories are updated simultaneously")
    print(f"  - Normalization ensures they sum to 1.0")
    print(f"  - This respects the constraint that exactly one diagnosis is correct")
    
    print(f"\n=== WHAT HAPPENS IF WE NORMALIZE THE ODDS RESULTS? ===")
    normalized_odds = odds_posteriors / np.sum(odds_posteriors)
    print(f"Normalized odds results: {normalized_odds}")
    print(f"Sum: {np.sum(normalized_odds):.6f}")
    
    print(f"\nComparison with probability form:")
    differences = normalized_odds - prob_posterior
    print(f"Differences: {differences}")
    print(f"Max difference: {np.max(np.abs(differences)):.6f}")
    
    if np.max(np.abs(differences)) < 0.001:
        print(f"✅ After normalization, both methods give nearly identical results")
    else:
        print(f"❌ Even after normalization, methods give different results")
    
    print(f"\n=== RECOMMENDATION ===")
    print(f"For multiple competing diagnostic categories:")
    print(f"1. PROBABILITY FORM is mathematically correct")
    print(f"   - Respects the constraint that probabilities sum to 1.0")
    print(f"   - Properly handles competing hypotheses")
    print(f"   - Standard approach in multi-class Bayesian inference")
    
    print(f"\n2. ODDS FORM is problematic for multiple categories")
    print(f"   - Treats categories independently")
    print(f"   - Results don't sum to 1.0 without additional normalization")
    print(f"   - Designed for binary comparisons (disease vs. no disease)")
    
    print(f"\n3. IF using odds form for multiple categories:")
    print(f"   - Must normalize results to sum to 1.0")
    print(f"   - This essentially converts it back to the probability form")
    print(f"   - Better to use probability form directly")
    
    return {
        'prob_form_sum': np.sum(prob_posterior),
        'odds_form_sum': np.sum(odds_posteriors),
        'prob_form_valid': abs(np.sum(prob_posterior) - 1.0) < 0.001,
        'odds_form_valid': abs(np.sum(odds_posteriors) - 1.0) < 0.001,
        'max_difference_after_norm': np.max(np.abs(differences))
    }

if __name__ == "__main__":
    results = compare_approaches_for_multi_category()
    
    print(f"\n=== FINAL VERDICT ===")
    if not results['odds_form_valid']:
        print(f"🎯 CLEAR RECOMMENDATION: Use PROBABILITY FORM for multiple categories")
        print(f"   - The spreadsheet's odds form is mathematically inappropriate")
        print(f"   - Your application's approach is correct")
        print(f"   - The spreadsheet should switch to probability form")
    else:
        print(f"Both approaches are valid, but probability form is preferred for multiple categories")