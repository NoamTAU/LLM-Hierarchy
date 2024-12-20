# LLM-Hierarchy
Probing the Latent Hierarchical Structure of Data via Large Language Models


*Tasks:*

1) Use the spacy to mask and potentially analyze the tree structure in generated unmasked text.
2) Use multiple models: LLama3.1 8B, Gemma 2 9B, LLama3.1 70B to get robust results.
3) Experiment with different masking tokens: \[mask\], \<mask\>, \<fill\>, \<unk\>, etc.
4) Experiment with different prompts and hyperparams.
5) Think of a good method to control generated length and quality, for instance adding \<start\> and \<end\> tokens that must be copied for a vaild reply, removing paragraphs with more than 2*\# masks extra words, etc.
6) Spin chain measurement: multiple ways to define the spin chain lengths, so far I'm using the fixed original text token length.
7) Save all the data for postprocessing.
