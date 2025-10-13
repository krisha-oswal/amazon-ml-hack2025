"""
MASTER SCRIPT: Run Complete Pipeline
Execute all steps automatically
"""

import subprocess
import sys
import time

def run_step(script_name, description, required=True):
    """Run a step and track time"""
    print("\n" + "="*80)
    print(f"🚀 {description}")
    print("="*80)
    
    start = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], check=True)
        elapsed = time.time() - start
        print(f"\n✅ {description} completed in {elapsed/60:.1f} minutes")
        return True
    except subprocess.CalledProcessError:
        elapsed = time.time() - start
        print(f"\n❌ {description} failed after {elapsed/60:.1f} minutes")
        if required:
            print("This is a required step. Please fix errors and try again.")
            return False
        return True
    except FileNotFoundError:
        print(f"\n❌ Script not found: {script_name}")
        return False

def main():
    print("="*80)
    print("AMAZON ML CHALLENGE - COMPLETE PIPELINE")
    print("Target: <10% SMAPE")
    print("="*80)
    
    pipeline_start = time.time()
    
    # Step 1: Data Analysis
    if not run_step('data_analysis_script.py', 'STEP 1: Data Analysis', required=True):
        return
    
    # Step 2: Feature Engineering
    if not run_step('feature_engineering_script.py', 'STEP 2: Feature Engineering', required=True):
        return
    
    # Step 3: Image Download 
    print("\n" + "="*80)
    print("STEP 3: Image Download ")
    print("="*80)
    print("This step takes 2-4 hours but improves SMAPE by 2-3%")
    response = input("Download images? (yes/no): ").strip().lower()
    
    if response == 'yes':
        run_step('src.py', 'STEP 3: Download Images', required=False)
        run_step('image_extract.py', 'STEP 3B: Extract Image Features', required=False)
    else:
        print("⏭️  Skipping image download (using text features only)")
    
    # Step 4: Model Training
    if not run_step('model_training.py', 'STEP 4: Model Training', required=True):
        return
    
    # Summary
    total_time = time.time() - pipeline_start
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"\n⏱️  Total time: {total_time/3600:.2f} hours")
    print("\n📁 Generated files:")
    print("  ✅ output/test_out.csv (SUBMISSION FILE)")
    print("  ✅ output/oof_predictions.csv")
    print("  ✅ output/model_scores.csv")
    print("  ✅ features/train_features_*.csv")
    print("  ✅ features/test_features_*.csv")
    
    print("\n📝 Next steps:")
    print("  1. Open output/model_scores.csv to see your CV SMAPE")
    print("  2. Verify output/test_out.csv format")
    print("  3. Update documentation.md with your scores")
    print("  4. Submit test_out.csv to the platform")
    
    print("\n🚀 Good luck with your submission!")

if __name__ == "__main__":
    main()