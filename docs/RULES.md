# Rules

- Your model must have **no more than 50,000,000 total trainable parameters**. This count **includes token embeddings and the output head**.
- Your repository must include a script or printed output showing the parameter count, plus your model config.
- Your model must be **trained from scratch**. You may not initialize from pretrained weights, fine-tune an existing model, or distill from a larger model.
- You **may use** standard frameworks and libraries (PyTorch, JAX, TensorFlow, Hugging Face Transformers, tokenizer libraries, and similar), and you **may use** existing public datasets (FineWeb, TinyStories, The Pile, and similar). “From scratch” refers to the weights and the training run, not to reinventing your tooling. Credit everything you use in Built With and your README.
- You **may not** use a hosted inference API, including the Featherless sponsor perk, as the model itself. Sponsor API access is for tooling, data processing, and evaluation, not as your submission.
- **Reporting:** your README must state the hardware you trained on, total training time, and approximate compute used. Training efficiency is a scored criterion and cannot be judged without this.
- **Evaluation:** projects are scored on HellaSwag, ARC-Easy, PIQA, and WinoGrande via `lm-evaluation-harness`, plus perplexity on a held-out slice of WikiText-103. Report your numbers in your README and include the evaluation script you used.
