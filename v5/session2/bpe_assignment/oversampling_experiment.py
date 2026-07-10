import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from step3_optimized import build_base_tokenizer, make_trainer, lines_from_text, preprocess
from utils import fetch_all_languages, compute_fertility

def compute_score(ratios):
    vals = list(ratios.values())
    spread = max(vals) - min(vals)
    return 1000.0 / spread if spread > 0 else float("inf")

def main():
    print("Loading data...")
    texts = fetch_all_languages()
    cleaned_texts = preprocess(texts)
    
    factors = [1, 2, 3, 5, 8, 10, 15, 20, 50, 100]
    
    print("\nOversampling Experiment Results:")
    print(f"{'Factor':<8} | {'Score':<10} | {'Spread':<8} | {'X_EN':<8} | {'X_HI':<8} | {'X_TE':<8} | {'X_KN':<8}")
    print("-" * 75)
    
    for factor in factors:
        tokenizer = build_base_tokenizer()
        trainer = make_trainer()
        
        def oversampled_iterator():
            # English 1x
            yield from lines_from_text(cleaned_texts["en"])
            # Indic Nx
            for lang in ["hi", "te", "kn"]:
                for _ in range(factor):
                    yield from lines_from_text(cleaned_texts[lang])
        
        tokenizer.train_from_iterator(oversampled_iterator(), trainer=trainer)
        
        ratios = {}
        for lang, text in cleaned_texts.items():
            ratios[lang] = compute_fertility(tokenizer, text)
            
        score = compute_score(ratios)
        spread = max(ratios.values()) - min(ratios.values())
        
        print(f"{factor:<8} | {score:<10.2f} | {spread:<8.4f} | {ratios['en']:<8.4f} | {ratios['hi']:<8.4f} | {ratios['te']:<8.4f} | {ratios['kn']:<8.4f}")

if __name__ == "__main__":
    main()
