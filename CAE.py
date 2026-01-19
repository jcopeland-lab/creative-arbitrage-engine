# %% [markdown]
# # The Revenue Refinery
# ## Predictive Creative Intelligence for Performance Marketing
# 
# **Goal:** Build a system that connects *what a creative looks like* (Creative DNA) to *how it performs economically* (Economic DNA), enabling predictive recommendations for future creative development.
# 
# **Dataset:** Sickness Pt.1 music campaign - 3 ad creatives across Meta Ads
# 
# **Key Questions:**
# 1. Which creative patterns drive the best CPR (Cost Per Result)?
# 2. What auction dynamics (CPM, CTR, Frequency) predict economic outcomes?
# 3. Can we score creatives as Unicorns, Tests, or Sick before scaling?
# 
# ---
# 
# ### The Architect's Glossary
# 
# | Term | Definition |
# |------|------------|
# | **Golden Handshake** | Joining Creative DNA (visual/audio features) to Economic DNA (performance metrics) |
# | **The Brain** | Baseline definitions - distributional truth (medians, quartiles) for all metrics |
# | **The Eyes** | Computer vision extraction - brightness, contrast, motion, cuts |
# | **The Ears** | Audio analysis - BPM, energy, spectral features |
# | **Unicorn** | Elite performer (top 25% on composite score) |
# | **Sick** | Underperformer (bottom 25% on composite score) |
# 
# ---

# %% [markdown]
# ## Section 1: Foundation

# %%
# Core imports
import duckdb
import pandas as pd
import numpy as np
import os

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# The Eyes (Computer Vision)
import cv2

# The Ears (Audio Analysis)
import librosa

# Statistics
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Settings
plt.style.use('seaborn-v0_8-whitegrid')
pd.set_option('display.float_format', lambda x: f'{x:,.3f}')

# Project paths
PROJECT_ROOT = '/Users/macbookpro/Desktop/Marketing Analytics/Spotify_Incrementality_Project'
DATA_PATH = os.path.join(PROJECT_ROOT, 'refinery_data', 'Granular Daily Ad Report.csv')
CREATIVEZ_PATH = os.path.join(PROJECT_ROOT, 'creativez')
DB_PATH = os.path.join(PROJECT_ROOT, 'refinery.db')

# Database connection
con = duckdb.connect(DB_PATH)

print("Environment ready.")
print(f"Data file: {DATA_PATH}")
print(f"Creativez folder: {CREATIVEZ_PATH}")

# %% [markdown]
# ## Section 2: Data Ingestion & Cleaning

# %%
# Ingest raw CSV with normalization
con.execute(f"""
    CREATE OR REPLACE TABLE raw_ads AS
    SELECT * FROM read_csv_auto(
        '{DATA_PATH}',
        normalize_names=true
    )
""")

# Check what columns we have
columns = con.execute("DESCRIBE raw_ads").df()
print("Available columns:")
print(columns['column_name'].tolist())

# %%
# Clean and structure the data
con.execute("""
    CREATE OR REPLACE TABLE ads AS
    SELECT
        ad_id,
        ad_name,
        campaign_name,
        CAST(day AS DATE) as date,
        
        -- Spend
        COALESCE(amount_spent_usd, 0) as amount_spent_usd,
        
        -- Volume
        COALESCE(results, 0) as results,
        
        -- Efficiency
        COALESCE(cost_per_result, 0) as cost_per_result,
        COALESCE(cpm_cost_per_1000_impressions, 0) as cpm,
        COALESCE(ctr_link_clickthrough_rate, 0) as ctr,
        COALESCE(cpc_cost_per_link_click, 0) as cpc,
        
        -- Engagement
        COALESCE(frequency, 0) as frequency,
        
        -- Retention (ThruPlays / 3-sec views)
        CASE 
            WHEN video_plays_at_3_seconds > 0 
            THEN (CAST(video_thru_plays AS FLOAT) / video_plays_at_3_seconds) * 100
            ELSE 0 
        END as retention_rate
        
    FROM raw_ads
    WHERE ad_id IS NOT NULL
""")

# Verify
row_count = con.execute("SELECT COUNT(*) FROM ads").fetchone()[0]
print(f"Cleaned ads table: {row_count} rows")

# %%
# Quick data overview
overview = con.execute("""
    SELECT 
        COUNT(DISTINCT ad_id) as unique_ads,
        COUNT(DISTINCT campaign_name) as unique_campaigns,
        MIN(date) as first_date,
        MAX(date) as last_date,
        SUM(amount_spent_usd) as total_spend,
        SUM(results) as total_results
    FROM ads
""").df()

print("DATA OVERVIEW")
print("="*50)
print(overview.T)

# %% [markdown]
# ## Section 3: Exploratory Data Analysis (The Vibe Check)

# %%
# Campaign-level summary
print("CAMPAIGN PERFORMANCE SUMMARY")
print("="*80)

campaign_summary = con.execute("""
    SELECT 
        campaign_name,
        COUNT(DISTINCT ad_id) as num_ads,
        COUNT(*) as data_points,
        SUM(amount_spent_usd) as total_spend,
        SUM(results) as total_results,
        MEDIAN(cost_per_result) as median_cpr,
        MEDIAN(ctr) as median_ctr,
        MEDIAN(retention_rate) as median_retention
    FROM ads
    WHERE amount_spent_usd > 0 AND results > 0
    GROUP BY campaign_name
    ORDER BY total_spend DESC
""").df()

campaign_summary

# %%
# Ad-level summary for our target campaign
print("AD PERFORMANCE: Sickness Pt.1 Campaign")
print("="*80)

ad_summary = con.execute("""
    SELECT 
        ad_name,
        COUNT(*) as days_active,
        SUM(amount_spent_usd) as total_spend,
        SUM(results) as total_results,
        MEDIAN(cost_per_result) as median_cpr,
        MEDIAN(ctr) as median_ctr,
        MEDIAN(retention_rate) as median_retention
    FROM ads
    WHERE campaign_name = 'Sickness Pt.1 | America/Canada'
    AND amount_spent_usd > 0 AND results > 0
    GROUP BY ad_name
    ORDER BY median_cpr ASC
""").df()

ad_summary

# %%
# Visualize: CPR distribution by ad
viz_data = con.execute("""
    SELECT ad_name, cost_per_result as cpr
    FROM ads
    WHERE campaign_name = 'Sickness Pt.1 | America/Canada'
    AND amount_spent_usd > 0 AND results > 0
""").df()

fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=viz_data, x='ad_name', y='cpr', ax=ax)
ax.set_title('Cost Per Result Distribution by Ad', fontsize=12, fontweight='bold')
ax.set_xlabel('Ad Name')
ax.set_ylabel('Cost Per Result ($)')
plt.xticks(rotation=15)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 4: The Brain (Baseline Definitions)
# 
# Establishing "The Truth" - distributional baselines using **MEDIAN** for performance metrics.

# %%
# Calculate distributional baselines
print("THE BRAIN: Distributional Baselines")
print("="*80)

baselines = con.execute("""
    SELECT 
        -- CPR (Cost Per Result)
        QUANTILE_CONT(cost_per_result, 0.25) AS p25_cpr,
        MEDIAN(cost_per_result) AS med_cpr,
        QUANTILE_CONT(cost_per_result, 0.75) AS p75_cpr,
        STDDEV(cost_per_result) AS std_cpr,
        
        -- Retention
        QUANTILE_CONT(retention_rate, 0.25) AS p25_retention,
        MEDIAN(retention_rate) AS med_retention,
        QUANTILE_CONT(retention_rate, 0.75) AS p75_retention,
        
        -- CTR
        QUANTILE_CONT(ctr, 0.25) AS p25_ctr,
        MEDIAN(ctr) AS med_ctr,
        QUANTILE_CONT(ctr, 0.75) AS p75_ctr,
        
        -- CPM
        QUANTILE_CONT(cpm, 0.25) AS p25_cpm,
        MEDIAN(cpm) AS med_cpm,
        QUANTILE_CONT(cpm, 0.75) AS p75_cpm,
        
        -- Volume
        QUANTILE_CONT(results, 0.25) AS p25_volume,
        MEDIAN(results) AS med_volume,
        QUANTILE_CONT(results, 0.75) AS p75_volume
        
    FROM ads
    WHERE campaign_name = 'Sickness Pt.1 | America/Canada'
    AND amount_spent_usd > 0 AND results > 0
""").df()

# Store as variables for use throughout notebook
p25_cpr = baselines['p25_cpr'][0]
med_cpr = baselines['med_cpr'][0]
p75_cpr = baselines['p75_cpr'][0]

p25_retention = baselines['p25_retention'][0]
med_retention = baselines['med_retention'][0]
p75_retention = baselines['p75_retention'][0]

print("CPR (Cost Per Result):")
print(f"  Elite (P25): < ${p25_cpr:.3f}")
print(f"  Normal (Median): ${med_cpr:.3f}")
print(f"  Sick (P75): > ${p75_cpr:.3f}")
print()
print("Retention Rate:")
print(f"  Sick (P25): < {p25_retention:.1f}%")
print(f"  Normal (Median): {med_retention:.1f}%")
print(f"  Elite (P75): > {p75_retention:.1f}%")

# %%
# Causal Inference: Testing correlations between auction inputs and economic outputs
print("THE BRAIN: Causal Inference")
print("="*80)
print("Testing: Do auction inputs (CPM, CTR, Frequency) predict economic outputs (CPR, Volume)?")
print()

causal_data = con.execute("""
    SELECT cpm, ctr, frequency, cost_per_result as cpr, results as volume, retention_rate
    FROM ads
    WHERE campaign_name = 'Sickness Pt.1 | America/Canada'
    AND amount_spent_usd > 0 AND results > 0
    AND retention_rate > 0
""").df().dropna()

# Test key relationships
relationships = [
    ('cpm', 'cpr', 'CPM → CPR'),
    ('ctr', 'cpr', 'CTR → CPR'),
    ('ctr', 'volume', 'CTR → Volume'),
    ('frequency', 'cpr', 'Frequency → CPR'),
]

print(f"{'Relationship':<20} {'Correlation':>12} {'P-Value':>12} {'Significant?':>12}")
print("-" * 60)

for x_col, y_col, label in relationships:
    corr, p_val = pearsonr(causal_data[x_col], causal_data[y_col])
    sig = "YES" if p_val < 0.05 else "NO"
    print(f"{label:<20} {corr:>12.3f} {p_val:>12.4f} {sig:>12}")

# %% [markdown]
# ## Section 5: The Eyes (Visual DNA Extraction)

# %%
def extract_visual_dna(video_path, analyze_first_n_seconds=3):
    """
    Extract visual features from a video file.
    
    Returns:
        avg_brightness: Mean brightness (0-255)
        avg_contrast: Mean contrast (standard deviation of pixel values)
        motion_intensity: Average frame-to-frame difference
        cuts_per_second: Estimated scene changes per second
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Could not open: {video_path}")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(fps * analyze_first_n_seconds)
    
    brightness_values = []
    contrast_values = []
    motion_scores = []
    scene_changes = 0
    prev_frame = None
    
    for i in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Brightness & Contrast
        brightness_values.append(np.mean(gray))
        contrast_values.append(np.std(gray))
        
        # Motion detection
        if prev_frame is not None:
            diff = cv2.absdiff(gray, prev_frame)
            motion_scores.append(np.mean(diff))
            
            # Scene change detection
            if np.mean(diff) > 30:
                scene_changes += 1
                
        prev_frame = gray
    
    cap.release()
    
    return {
        'avg_brightness': np.mean(brightness_values) if brightness_values else 0,
        'avg_contrast': np.mean(contrast_values) if contrast_values else 0,
        'motion_intensity': np.mean(motion_scores) if motion_scores else 0,
        'cuts_per_second': scene_changes / analyze_first_n_seconds
    }

print("Visual DNA extractor ready.")

# %% [markdown]
# ## Section 6: The Ears (Audio DNA Extraction)

# %%
def extract_audio_dna(video_path, analyze_first_n_seconds=3):
    """
    Extract audio features from a video file.
    
    Returns:
        bpm: Beats per minute (tempo)
        spectral_centroid: "Brightness" of sound (higher = brighter/sharper)
        rms_energy: Loudness/energy level
        audio_intensity: Categorical (Low/Medium/High)
    """
    try:
        y, sr = librosa.load(video_path, duration=analyze_first_n_seconds)
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        
        # Spectral centroid (brightness of sound)
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # RMS energy (loudness)
        rms = librosa.feature.rms(y=y)[0]
        
        # Classify intensity
        avg_rms = np.mean(rms)
        if avg_rms > 0.1:
            intensity = 'High'
        elif avg_rms > 0.05:
            intensity = 'Medium'
        else:
            intensity = 'Low'
        
        return {
            'bpm': float(tempo) if np.isscalar(tempo) else float(tempo[0]),
            'spectral_centroid': float(np.mean(spectral_centroids)),
            'rms_energy': float(avg_rms),
            'audio_intensity': intensity
        }
        
    except Exception as e:
        print(f"Audio extraction failed for {video_path}: {e}")
        return None

print("Audio DNA extractor ready.")

# %% [markdown]
# ## Section 7: Processing Creatives

# %%
# Map video files to ad IDs
creative_mapping = {
    '120218938658560104': 'Sickness pt.1 | Train.MP4',
    '120218884405900104': 'Sickness pt.1 | Train 2.MP4',
    '120218938658570104': 'Sickness pt.1 | Train 3.MP4',
}

# Extract DNA from all creatives
print("EXTRACTING CREATIVE DNA")
print("="*80)

creative_dna_records = []

for ad_id, video_file in creative_mapping.items():
    full_path = os.path.join(CREATIVEZ_PATH, video_file)
    
    if not os.path.exists(full_path):
        print(f"File not found: {full_path}")
        continue
    
    print(f"Processing: {video_file}")
    
    # Extract visual DNA
    visual = extract_visual_dna(full_path)
    
    # Extract audio DNA
    audio = extract_audio_dna(full_path)
    
    if visual and audio:
        record = {
            'ad_id': ad_id,
            **visual,
            **audio
        }
        creative_dna_records.append(record)
        print(f"  Visual: brightness={visual['avg_brightness']:.1f}, motion={visual['motion_intensity']:.1f}")
        print(f"  Audio: BPM={audio['bpm']:.0f}, intensity={audio['audio_intensity']}")
    print()

# Store in database
if creative_dna_records:
    creative_dna_df = pd.DataFrame(creative_dna_records)
    con.execute("CREATE OR REPLACE TABLE creative_dna_extracted AS SELECT * FROM creative_dna_df")
    print(f"Stored {len(creative_dna_records)} creative DNA profiles.")

# %% [markdown]
# ## Section 8: Creative DNA Translation (Tech → Actionable)
# 
# Converting technical metrics to human-readable creative briefs.

# %%
def translate_creative_dna(row):
    """
    Convert technical metrics to actionable creative language.
    """
    translations = {}
    
    # Motion style
    motion = row['motion_intensity']
    if motion > 10:
        translations['motion_style'] = "Fast-paced, High-energy movement"
    elif motion > 5:
        translations['motion_style'] = "Moderate movement, Dynamic"
    else:
        translations['motion_style'] = "Slow, Static shots"
    
    # Lighting style
    brightness = row['avg_brightness']
    if brightness > 150:
        translations['lighting_style'] = "Bright, Well-lit"
    elif brightness > 100:
        translations['lighting_style'] = "Balanced lighting"
    else:
        translations['lighting_style'] = "Dark, Moody"
    
    # Edit pace
    cuts = row['cuts_per_second']
    if cuts > 1:
        translations['edit_pace'] = "Rapid cuts (MTV style)"
    elif cuts > 0.3:
        translations['edit_pace'] = "Medium pace editing"
    else:
        translations['edit_pace'] = "Long takes, Minimal cuts"
    
    # Audio energy
    translations['audio_energy'] = f"{row['audio_intensity']} energy audio"
    translations['tempo'] = f"{row['bpm']:.0f} BPM"
    
    return translations

# Apply translations
print("CREATIVE DNA TRANSLATION")
print("="*80)

creative_dna_df = con.execute("SELECT * FROM creative_dna_extracted").df()

for idx, row in creative_dna_df.iterrows():
    trans = translate_creative_dna(row)
    ad_name = con.execute(f"SELECT DISTINCT ad_name FROM ads WHERE ad_id = '{row['ad_id']}'").fetchone()[0]
    
    print(f"\n{ad_name}:")
    for key, value in trans.items():
        print(f"  {key}: {value}")

# %% [markdown]
# ## Section 9: The Golden Handshake
# 
# Joining Creative DNA to Economic DNA - the core of the Refinery.

# %%
# Create the Golden Handshake table
con.execute("""
    CREATE OR REPLACE TABLE golden_handshake AS
    SELECT 
        a.ad_id,
        a.ad_name,
        a.date,
        
        -- Creative DNA (The Eyes)
        c.avg_brightness,
        c.avg_contrast,
        c.motion_intensity,
        c.cuts_per_second,
        
        -- Creative DNA (The Ears)
        c.bpm,
        c.spectral_centroid,
        c.rms_energy,
        
        -- Economic DNA (Auction Inputs)
        a.cpm,
        a.ctr,
        a.cpc,
        a.frequency,
        
        -- Economic DNA (Outputs)
        a.cost_per_result as cpr,
        a.results as volume,
        a.retention_rate,
        a.amount_spent_usd as spend
        
    FROM ads a
    LEFT JOIN creative_dna_extracted c ON a.ad_id = c.ad_id
    WHERE a.campaign_name = 'Sickness Pt.1 | America/Canada'
    AND a.amount_spent_usd > 0
    AND a.results > 0
""")

# Verify
handshake_count = con.execute("SELECT COUNT(*) FROM golden_handshake").fetchone()[0]
print(f"Golden Handshake table: {handshake_count} rows")
print()
con.execute("SELECT * FROM golden_handshake LIMIT 3").df()

# %% [markdown]
# ## Section 10: Unicorn Scoring Matrix
# 
# Composite scoring to classify ads as Unicorn / Test / Sick.

# %%
# Calculate Unicorn scores
# Weights: 40% CPR, 30% Retention, 20% Volume, 10% CTR

print("UNICORN SCORING MATRIX")
print("="*80)

unicorn_scores = con.execute("""
    WITH ad_metrics AS (
        SELECT 
            ad_name,
            MEDIAN(cpr) as median_cpr,
            MEDIAN(retention_rate) as median_retention,
            SUM(volume) as total_volume,
            MEDIAN(ctr) as median_ctr
        FROM golden_handshake
        GROUP BY ad_name
    ),
    normalized AS (
        SELECT 
            ad_name,
            median_cpr,
            median_retention,
            total_volume,
            median_ctr,
            
            -- Normalize to 0-100 scale (lower CPR = better)
            (1 - (median_cpr - MIN(median_cpr) OVER()) / 
                 NULLIF(MAX(median_cpr) OVER() - MIN(median_cpr) OVER(), 0)) * 100 as cpr_score,
            
            -- Higher retention = better
            (median_retention - MIN(median_retention) OVER()) / 
                NULLIF(MAX(median_retention) OVER() - MIN(median_retention) OVER(), 0) * 100 as retention_score,
            
            -- Higher volume = better
            (total_volume - MIN(total_volume) OVER()) / 
                NULLIF(MAX(total_volume) OVER() - MIN(total_volume) OVER(), 0) * 100 as volume_score,
            
            -- Higher CTR = better
            (median_ctr - MIN(median_ctr) OVER()) / 
                NULLIF(MAX(median_ctr) OVER() - MIN(median_ctr) OVER(), 0) * 100 as ctr_score
        FROM ad_metrics
    )
    SELECT 
        ad_name,
        median_cpr,
        median_retention,
        total_volume,
        
        -- Weighted composite score
        COALESCE(cpr_score, 50) * 0.4 + 
        COALESCE(retention_score, 50) * 0.3 + 
        COALESCE(volume_score, 50) * 0.2 + 
        COALESCE(ctr_score, 50) * 0.1 as unicorn_score,
        
        -- Classification
        CASE 
            WHEN (COALESCE(cpr_score, 50) * 0.4 + 
                  COALESCE(retention_score, 50) * 0.3 + 
                  COALESCE(volume_score, 50) * 0.2 + 
                  COALESCE(ctr_score, 50) * 0.1) >= 75 THEN 'UNICORN - Scale'
            WHEN (COALESCE(cpr_score, 50) * 0.4 + 
                  COALESCE(retention_score, 50) * 0.3 + 
                  COALESCE(volume_score, 50) * 0.2 + 
                  COALESCE(ctr_score, 50) * 0.1) >= 50 THEN 'TEST - Monitor'
            ELSE 'SICK - Kill'
        END as classification
    FROM normalized
    ORDER BY unicorn_score DESC
""").df()

unicorn_scores

# %% [markdown]
# ## Section 11: Strategic Correlation Matrix
# 
# The "First Three → Last Two" framework: How do auction inputs predict economic outputs?

# %%
# Build correlation matrix
print("STRATEGIC CORRELATION MATRIX")
print("="*80)

matrix_data = con.execute("""
    SELECT cpm, ctr, cpc, frequency, cpr, volume, retention_rate,
           motion_intensity, avg_brightness, bpm
    FROM golden_handshake
    WHERE cpr > 0 AND retention_rate > 0
""").df().dropna()

# Calculate correlation matrix
corr_matrix = matrix_data.corr()

# Visualize
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
            square=True, ax=ax)
ax.set_title('Strategic Correlation Matrix: Auction + Creative DNA → Economic Outcomes', 
             fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
# The 4 Critical Correlations (per mentor request)
print("THE 4 CRITICAL CORRELATIONS")
print("="*80)

critical_pairs = [
    ('ctr', 'cpr', 'CTR → CPR'),
    ('ctr', 'volume', 'CTR → Volume'),
    ('cpm', 'cpr', 'CPM → CPR'),
    ('cpm', 'volume', 'CPM → Volume'),
]

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

for idx, (x_col, y_col, title) in enumerate(critical_pairs):
    ax = axes[idx // 2, idx % 2]
    
    x = matrix_data[x_col]
    y = matrix_data[y_col]
    
    # Scatter plot
    ax.scatter(x, y, alpha=0.5)
    
    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(x.sort_values(), p(x.sort_values()), "r--", alpha=0.8)
    
    # Stats
    corr, p_val = pearsonr(x, y)
    
    ax.set_title(f'{title}\nr={corr:.3f}, p={p_val:.4f}', fontsize=11, fontweight='bold')
    ax.set_xlabel(x_col.upper())
    ax.set_ylabel(y_col.upper())

plt.tight_layout()
plt.show()

# %% [markdown]
# ## Section 12: Business Questions

# %%
# Business Question #1: Which Creative DNA patterns drive best CPR?
print("BUSINESS QUESTION #1: Which Creative DNA Patterns Drive Best CPR?")
print("="*80)

dna_performance = con.execute("""
    SELECT 
        ad_name,
        AVG(motion_intensity) as avg_motion,
        AVG(avg_brightness) as avg_brightness,
        AVG(avg_contrast) as avg_contrast,
        AVG(cuts_per_second) as avg_cuts,
        AVG(bpm) as avg_bpm,
        AVG(rms_energy) as avg_energy,
        MEDIAN(cpr) as median_cpr,
        MEDIAN(retention_rate) as median_retention,
        SUM(volume) as total_volume
    FROM golden_handshake
    GROUP BY ad_name
    ORDER BY median_cpr ASC
""").df()

print("\nRanked by Median CPR (best to worst):")
print(dna_performance[['ad_name', 'median_cpr', 'median_retention', 'total_volume']].to_string(index=False))

# Winner analysis
best = dna_performance.iloc[0]
worst = dna_performance.iloc[-1]

print(f"\n" + "="*80)
print(f"WINNER: {best['ad_name']}")
print(f"  Median CPR: ${best['median_cpr']:.3f}")
print(f"  Median Retention: {best['median_retention']:.1f}%")
print(f"  Motion Intensity: {best['avg_motion']:.1f}")
print(f"  Brightness: {best['avg_brightness']:.1f}")

cpr_improvement = ((worst['median_cpr'] - best['median_cpr']) / worst['median_cpr'] * 100)
print(f"\nCPR Advantage: {cpr_improvement:.1f}% better than worst performer")

# %%
# Statistical Significance Check: Did you kill the wrong ad?
print("STATISTICAL SIGNIFICANCE CHECK: Sample Size Analysis")
print("="*80)

sample_analysis = con.execute("""
    SELECT 
        ad_name,
        COUNT(*) as days_active,
        MIN(date) as first_seen,
        MAX(date) as last_seen,
        MEDIAN(cpr) as median_cpr,
        QUANTILE_CONT(cpr, 0.25) as p25_cpr,
        QUANTILE_CONT(cpr, 0.75) as p75_cpr,
        STDDEV(cpr) as cpr_volatility,
        SUM(volume) as total_volume,
        SUM(spend) as total_spend
    FROM golden_handshake
    GROUP BY ad_name
    ORDER BY median_cpr ASC
""").df()

print(sample_analysis.to_string(index=False))

# Check for sample size issues
print("\n" + "="*80)
winner = sample_analysis.iloc[0]
runner_up = sample_analysis.iloc[1]

if winner['days_active'] < runner_up['days_active'] / 2:
    print("WARNING: Winner has significantly smaller sample size!")
    print(f"  {winner['ad_name']}: {winner['days_active']} days")
    print(f"  {runner_up['ad_name']}: {runner_up['days_active']} days")
    print("  The ad you killed might have been killed too early.")
else:
    print("Sample sizes are comparable - results are statistically sound.")

# %% [markdown]
# ## Section 13: Executive Summary

# %%
print("="*80)
print("EXECUTIVE SUMMARY: THE REVENUE REFINERY")
print("="*80)
print()

print("1. BASELINE DEFINITIONS (The Brain)")
print("-" * 40)
print(f"   Elite CPR: < ${p25_cpr:.3f}")
print(f"   Normal CPR: ${med_cpr:.3f}")
print(f"   Sick CPR: > ${p75_cpr:.3f}")
print()

print("2. CREATIVE DNA INSIGHTS")
print("-" * 40)
print(f"   Best Performer: {best['ad_name']}")
print(f"   Key Pattern: Motion={best['avg_motion']:.1f}, Brightness={best['avg_brightness']:.1f}")
print(f"   NOTE: All ads use same audio track - audio DNA has limited variance")
print()

print("3. UNICORN CLASSIFICATIONS")
print("-" * 40)
for _, row in unicorn_scores.iterrows():
    print(f"   {row['ad_name']}: {row['classification']} (Score: {row['unicorn_score']:.1f})")
print()

print("4. KEY CORRELATIONS")
print("-" * 40)
for x_col, y_col, label in critical_pairs:
    corr, p_val = pearsonr(matrix_data[x_col], matrix_data[y_col])
    sig = "*" if p_val < 0.05 else ""
    print(f"   {label}: r={corr:.3f} {sig}")
print("   (* = statistically significant at p<0.05)")
print()

print("5. RECOMMENDED ACTIONS")
print("-" * 40)
print(f"   - Scale: {unicorn_scores[unicorn_scores['classification'].str.contains('UNICORN')]['ad_name'].tolist()}")
print(f"   - Test: {unicorn_scores[unicorn_scores['classification'].str.contains('TEST')]['ad_name'].tolist()}")
print(f"   - Kill: {unicorn_scores[unicorn_scores['classification'].str.contains('SICK')]['ad_name'].tolist()}")
print()

print("6. NEXT STEPS")
print("-" * 40)
print("   - Test with different audio tracks to enable audio DNA analysis")
print("   - Add ROAS data when available")
print("   - Expand to more campaigns for cross-campaign insights")

# %%
# Close database connection
con.close()
print("Database connection closed.")


