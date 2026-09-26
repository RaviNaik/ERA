window.SESSION_DATA = {
  "heroStats": [
    {
      "value": "20M",
      "label": "parameters"
    },
    {
      "value": "50M",
      "label": "tokens per run"
    },
    {
      "value": "euler",
      "label": "selected reversible variant"
    },
    {
      "value": "1024",
      "label": "largest reversible batch trained"
    }
  ],
  "runs": [
    {
      "label": "exp1_baseline_x",
      "variant": "baseline",
      "status": "ok",
      "batch_size": 128,
      "total_steps": 763,
      "final_val_loss": 4.827873825292568,
      "final_val_accuracy": 0.2674898938782651,
      "final_val_perplexity": 124.94502309217096,
      "mean_tokens_per_sec": 154833.54578393555,
      "mfu": 0.13384685486876027,
      "peak_step_memory_mb": 9672.25732421875,
      "activation_saved_mb_per_sample": 41.820343017578125,
      "wall_clock_s": 354.5626064352691,
      "tokens_seen": 50003968
    },
    {
      "label": "exp2_midpoint_x",
      "variant": "midpoint",
      "status": "ok",
      "batch_size": 128,
      "total_steps": 763,
      "final_val_loss": 4.841973671922938,
      "final_val_accuracy": 0.2646183587564825,
      "final_val_perplexity": 126.71920722401639,
      "mean_tokens_per_sec": 126371.40262061486,
      "mfu": 0.10924257208270968,
      "peak_step_memory_mb": 4716.33642578125,
      "activation_saved_mb_per_sample": 3.003936767578125,
      "wall_clock_s": 423.9528220407665,
      "tokens_seen": 50003968
    },
    {
      "label": "exp3_euler_x",
      "variant": "euler",
      "status": "ok",
      "batch_size": 128,
      "total_steps": 763,
      "final_val_loss": 4.774483885363631,
      "final_val_accuracy": 0.2722223115285564,
      "final_val_perplexity": 118.44916549412387,
      "mean_tokens_per_sec": 115041.44116155374,
      "mfu": 0.09944831400122267,
      "peak_step_memory_mb": 4716.33642578125,
      "activation_saved_mb_per_sample": 3.003936767578125,
      "wall_clock_s": 464.3517103269696,
      "tokens_seen": 50003968
    },
    {
      "label": "exp4_euler_2x",
      "variant": "euler",
      "status": "ok",
      "batch_size": 256,
      "total_steps": 382,
      "final_val_loss": 5.224083642940012,
      "final_val_accuracy": 0.23396191597596822,
      "final_val_perplexity": 185.6909333411005,
      "mean_tokens_per_sec": 120363.5456251104,
      "mfu": 0.10404904144774194,
      "peak_step_memory_mb": 5165.83642578125,
      "activation_saved_mb_per_sample": 3.0039215087890625,
      "wall_clock_s": 439.9307506456971,
      "tokens_seen": 50069504
    },
    {
      "label": "exp5_euler_4x",
      "variant": "euler",
      "status": "ok",
      "batch_size": 512,
      "total_steps": 191,
      "final_val_loss": 6.174924324670122,
      "final_val_accuracy": 0.16802489708581256,
      "final_val_perplexity": 480.546657017896,
      "mean_tokens_per_sec": 121519.71659286109,
      "mfu": 0.10504850087974352,
      "peak_step_memory_mb": 8544.07861328125,
      "activation_saved_mb_per_sample": 3.0039138793945312,
      "wall_clock_s": 433.69001481309533,
      "tokens_seen": 50069504
    },
    {
      "label": "exp6_euler_8x",
      "variant": "euler",
      "status": "ok",
      "batch_size": 1024,
      "total_steps": 96,
      "final_val_loss": 6.746235183621824,
      "final_val_accuracy": 0.12223674935237093,
      "final_val_perplexity": 850.8494331676441,
      "mean_tokens_per_sec": 122599.48651906117,
      "mfu": 0.10598191493980405,
      "peak_step_memory_mb": 16750.07861328125,
      "activation_saved_mb_per_sample": 3.0039100646972656,
      "wall_clock_s": 430.9627169892192,
      "tokens_seen": 50331648
    }
  ],
  "findings": [
    "Selected reversible variant: euler.",
    "Lowest validation loss: exp3_euler_x (4.7745).",
    "Highest throughput: exp1_baseline_x (154,834 tokens/s).",
    "Smallest activation memory: exp6_euler_8x (3.00 MB per sequence)."
  ]
};
