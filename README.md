# Enhanced Transcript Feature Analysis with Belief Updating

This application processes clinical transcripts to assess features and perform sophisticated belief updating calculations using Bayesian inference, KL divergence, and entropy reduction analysis.

## Features

### Core Functionality
- **Feature Assessment**: Uses LLM to determine which clinical features are addressed in transcripts (binary output: 1/0)
- **Belief Updating**: Sequential Bayesian inference using likelihood ratios
- **Information Theory**: Calculates KL divergence and entropy reduction for each feature
- **Enhanced Reports**: Comprehensive DOCX reports with mathematical calculations

### Mathematical Calculations
- **Entropy Calculation**: -Σ(p * log₂(p)) using log base 2
- **KL Divergence**: Σ(q * log₂(q/p)) for belief updating
- **Sequential Processing**: Features processed in order with cumulative statistics
- **Belief Updating**: Prior × LR → normalize for posterior probabilities

## Workflow

### Step 1: Upload Feature-LR Matrix
Upload a CSV or Excel file with:
- **First column**: Clinical features (e.g., "Patient Has: food gets stuck")
- **Remaining columns**: Diagnostic categories with likelihood ratio values

### Step 2: Set Prior Probabilities
Enter prior probabilities for each diagnostic category. Values must sum to 1.0.

### Step 3: Upload Transcripts
Upload PDF transcript files for analysis.

### Step 4: Generate Reports
The system will:
1. Extract text from PDFs
2. Use LLM to assess which features are addressed (1/0)
3. Apply likelihood ratios for addressed features
4. Perform sequential belief updating
5. Calculate KL divergence and entropy reduction
6. Generate enhanced DOCX reports

## Output Reports

Each DOCX report includes:

### 1. Feature Assessment Summary
- Total features assessed
- Features addressed vs. not addressed
- Percentage addressed

### 2. Complete Feature Assessment
- All features with binary results (1/0)

### 3. Addressed Features with Likelihood Ratios
- Features that were addressed
- LR values for each diagnostic category

### 4. Belief Updating Sequence
- Step-by-step probability updates
- Entropy before/after each feature
- Entropy reduction per feature
- KL divergence per feature

### 5. Final Diagnostic Probabilities
- Updated probabilities for each diagnostic category

### 6. Summary Statistics
- Initial and final entropy
- Total entropy reduction
- Cumulative KL divergence
- Number of belief updates performed

## Sample Data

The repository includes:
- `sample_feature_lr_matrix.csv`: Example feature-LR matrix with 78 clinical features
- `sample_transcript.txt`: Example clinical transcript for testing

## Requirements

```
streamlit
pandas
PyPDF2
python-docx
openpyxl
openai
numpy
```

## Installation

1. Clone the repository
2. Install requirements: `pip install -r requirements.txt`
3. Set up secrets in `.streamlit/secrets.toml`:
   ```toml
   PASSWORD="your-password"
   OPENROUTER_API_KEY="your-openrouter-api-key"
   ```
4. Run the app: `streamlit run app.py`

## Mathematical Background

### Belief Updating Process
1. **Prior Probabilities**: Initial diagnostic category probabilities (π)
2. **Likelihood Ratios**: Evidence strength for each feature (LR)
3. **Unnormalized Posterior**: π × LR
4. **Normalization**: Divide by sum to ensure probabilities sum to 1
5. **Posterior Probabilities**: Updated beliefs (q)

### Information Theory Metrics
- **Entropy**: Measures uncertainty in probability distribution
- **KL Divergence**: Measures information gained from belief updating
- **Entropy Reduction**: Decrease in uncertainty after evidence

### Sequential Processing
Features are processed one at a time, with each posterior becoming the prior for the next feature, allowing cumulative belief updating throughout the clinical interview.

## Supported LLM Models

- Google Gemini 2.5 Flash/Pro
- OpenAI GPT-4.1/GPT-4.1-mini
- Anthropic Claude 3.7 Sonnet

## Error Handling

The application includes robust error handling for:
- Invalid file formats
- Mathematical edge cases (log of zero)
- Prior probability validation
- LLM response parsing
- File processing errors

## Use Cases

- Medical education assessment
- Clinical reasoning analysis
- Diagnostic probability tracking
- Information theory research
- Bayesian inference demonstration