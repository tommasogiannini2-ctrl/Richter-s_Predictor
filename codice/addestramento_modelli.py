from evaluation import ModelEvaluator



evaluator = ModelEvaluator(
    y_true=y_val,
    y_pred=y_pred,
    y_score=y_proba,        
    output_dir="output/grafici"
)
risultati = evaluator.valuta_tutto()