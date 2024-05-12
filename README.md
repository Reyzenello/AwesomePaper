# AwesomePaper
One of the best ML papers which we have reviewed on Agora and myself :)


Top ML Papers of the Week (April 29 - May 5) - 2024


Paper	Links	
1) Kolmogorov-Arnold Networks - proposes Kolmogorov-Arnold Networks (KANs) as alternatives to Multi-Layer Perceptrons (MLPs); KANs apply learnable activation functions on edges that represent the weights; with no linear weights used, KANs can outperform MLPs and possess faster neural scaling laws; the authors show that KANs can be used as collaborators to help scientists discover mathematics and physical laws.	Paper, Tweet	
2) Better and Faster LLMs via Multi-token Prediction - proposes a multi-token prediction approach that performs language modeling by training the predict the following n tokens using n independent output heads; the output heads operate on top of a shared transformer trunk; multi-token prediction is shown to be useful when using larger model sizes and can speed up inference up to 3x; the proposed 13B parameter models solves 12 % more problems on HumanEval and 17 % more on MBPP than comparable next-token models.	Paper, Tweet	
3) Med-Gemini - presents a family of multimodal models specialized in medicines and based on the strong multimodal and long-context reasoning capabilities of Gemini; achieves state-of-the-art performance on 10/14 benchmarks surpassing GPT-4 models; it achieves 91% accuracy on MedQA (USMLE) benchmark using an uncertainty-guided search strategy.	Paper, Tweet	
4) When to Retrieve? - presents an approach to train LLMs to effectively utilize information retrieval; it first proposes a training approach to teach an LLM to generate a special token, , when it's not confident or doesn't know the answer to a question; the fine-tuned model outperforms a base LLM in two fixed alternate settings that include never retrieving and always retrieving context	Paper, Tweet	
5) A Survey on Retrieval-Augmented Language Models - covers the most important recent developments in RAG and RAU systems; it includes evolution, taxonomy, and an analysis of applications; there is also a section on how to enhance different components of these systems and how to properly evaluate them; it concludes with a section on limitations and future directions.	Paper, Tweet	
6) An Open-source LM Specialized in Evaluating Other LMs - open-source Prometheus 2 (7B & 8x7B), state-of-the-art open evaluator LLMs that closely mirror human and GPT-4 judgments; they support both direct assessments and pair-wise ranking formats grouped with user-defined evaluation criteria; according to the experimental results, this open-source model seems to be the strongest among all open-evaluator LLMs; the key seems to be in merging evaluator LMs trained on either direct assessment or pairwise ranking formats.	Paper, Tweet	
7) Self-Play Preference Optimization - proposes a self-play-based method for aligning language models; this optimation procedure treats the problem as a constant-sum two-player game to identify the Nash equilibrium policy; it addresses the shortcomings of DPO and IPO and effectively increases the log-likelihood of chose responses and decreases the rejected ones; SPPO outperforms DPO and IPO on MT-Bench and the Open LLM Leaderboard.	Paper, Tweet	
8) Inner Workings of Transformer Language Models - presents a technical introduction to current techniques used to interpret the inner workings of Transformer-based language models; it provides a detailed overview of the internal mechanisms implemented in these models.	Paper, Tweet	
9) Multimodal LLM Hallucinations - provides an overview of the recent advances in identifying, evaluating, and mitigating hallucination in multimodal LLMs; it also provides an overview of causes, evaluation benchmarks, metrics, and other strategies to deal with challenges related to detecting hallucinations.	Paper, Tweet	
10) In-Context Learning with Long-Context Models - studies the behavior in-context learning of LLMs at extreme context lengths with long-context models; shows that performance increases as hundreds or thousands of demonstrations are used; demonstrates that long-context ICL is less sensitive to random input shuffling than short-context ICL; concludes that the effectiveness of long-context LLMs is not due to task learning but from attending to similar examples.	Paper, Tweet	
Top ML Papers of the Week (April 22 - April 28) - 2024


Paper	Links	
1) Phi-3 - a new 3.8B parameter language model called phi-3-mini trained on 3.3 trillion tokens and is reported to rival Mixtral 8x7B and GPT-3.5; has a default context length of 4K but also includes a version that is extended to 128K (phi-mini-128K); combines heavily filtered web data and synthetic data to train the 3.8B models; it also reports results on 7B and 14B models trained on 4.8T tokens (phi-3-small and phi-3-medium)	Paper, Tweet	
2) OpenELM - a new open language model that employs a layer-wise scaling strategy to efficiently allocate parameters and leading to better efficiency and accuracy; comes with different sizes such as 270M, 450M, 1.1B, and 3B; achieves a 2.36% improvement in accuracy compared to OLMo while requiring 2× fewer pre-training tokens.	Paper, Tweet	
3) Arctic - an open-source LLM (Apache 2.0 license.) that uses a unique Dense-MoE Hybrid transformer architecture; performs on par with Llama3 70B in enterprise metrics like coding (HumanEval+ & MBPP+), SQL (Spider) and instruction following (IFEval); claims to use 17x less compute budget than Llama 3 70B; the training compute is roughly under $2 million (less than 3K GPU weeks).	Paper, Tweet	
4) Make Your LLM Fully Utilize the Context - presents an approach to overcome the lost-in-the-middle challenge common in LLMs. It applies an explicit "information-intensive" training procedure on Mistral-7B to enable the LLM to fully utilize the context. It leverages a synthetic dataset where the answer requires fine-grained information awareness on a short segment (∼128 tokens) within a synthesized long context (4K−32K tokens), and 2) the integration and reasoning of information from two or more short segments. The resulting model, FILM-7B (Fill-in-the-Middle), shows that it can robustly retrieve information from different positions in its 32K context window.	Paper, Tweet	
5) FineWeb - a large-scale web dataset containing 15 trillion tokens for training language models; filters and deduplicates CommonCrawl between 2013 and 2024 and the goal is to improve the quality of the data.	Paper, Tweet	
6) AI-powered Gene Editors - achieves precision editing of the human genome with a programmable gene editor design with an AI system powered by an LLM trained on biological diversity at scale.	Paper, Tweet	
7) AutoCrawler - Combines LLMs with crawlers with the goal of helping crawlers handle diverse and changing web environments more efficiently; the web crawler agent leverages the hierarchical structure of HTML for progressive understanding; employs top-down and step-back operations, and leverages the DOM tree structure, to generate a complete and executable crawler.	Paper, Tweet	
8) Graph Machine Learning in the Era of LLMs - provides a comprehensive overview of the latest advancements for Graph ML in the era of LLMs; covers the recent developments in Graph ML, how LLM can enhance graph features, and how it can address issues such as OOD and graph heterogeneity.	Paper, Tweet	
9) Self-Evolution of LLMs - provides a comprehensive survey on self-evolution approaches in LLMs.	Paper, Tweet	
10) Naturalized Execution Tuning (NExT) - trains an LLM to have the ability to inspect the execution traced of programs and reason about run-time behavior via synthetic chain-of-thought rationales; improves the fix rate of a PaLM 2 model on MBPP and Human by 26.1% and 14.3%; the model also shows that it can generalize to unknown scenarios.	Paper, Tweet	
Top ML Papers of the Week (April 15 - April 21) - 2024


Paper	Links	
1) Llama 3 - a family of LLMs that include 8B and 70B pretrained and instruction-tuned models; Llama 3 8B outperforms Gemma 7B and Mistral 7B Instruct; Llama 3 70 broadly outperforms Gemini Pro 1.5 and Claude 3 Sonnet.	Paper, Tweet	
2) Mixtral 8x22B - a new open-source sparse mixture-of-experts model that reports that compared to the other community models, it delivers the best performance/cost ratio on MMLU; shows strong performance on reasoning, knowledge retrieval, maths, and coding.	Paper, Tweet	
3) Chinchilla Scaling: A replication attempt - attempts to replicate the third estimation procedure of the compute-optimal scaling law proposed in Hoffmann et al. (2022) (i.e., Chinchilla scaling); finds that “the reported estimates are inconsistent with their first two estimation methods, fail at fitting the extracted data, and report implausibly narrow confidence intervals.”	Paper, Tweet	
4) How Faithful are RAG Models? - aims to quantify the tug-of-war between RAG and LLMs' internal prior; it focuses on GPT-4 and other LLMs on question answering for the analysis; finds that providing correct retrieved information fixes most of the model mistakes (94% accuracy); when the documents contain more incorrect values and the LLM's internal prior is weak, the LLM is more likely to recite incorrect information; the LLMs are found to be more resistant when they have a stronger prior.	Paper, Tweet	
5) A Survey on Retrieval-Augmented Text Generation for LLMs - presents a comprehensive overview of the RAG domain, its evolution, and challenges; it includes a detailed discussion of four important aspects of RAG systems: pre-retrieval, retrieval, post-retrieval, and generation.	Paper, Tweet	
6) The Illusion of State in State-Space Models - investigates the expressive power of state space models (SSMs) and reveals that it is limited similar to transformers in that SSMs cannot express computation outside the complexity class 𝖳𝖢^0; finds that SSMs cannot solve state-tracking problems like permutation composition and other tasks such as evaluating code or tracking entities in a long narrative.	Paper, Tweet	
7) Reducing Hallucination in Structured Outputs via RAG - discusses how to deploy an efficient RAG system for structured output tasks; the RAG system combines a small language model with a very small retriever; it shows that RAG can enable deploying powerful LLM-powered systems in limited-resource settings while mitigating issues like hallucination and increasing the reliability of outputs.	Paper, Tweet	
8) Emerging AI Agent Architectures - presents a concise summary of emerging AI agent architectures; it focuses the discussion on capabilities like reasoning, planning, and tool calling which are all needed to build complex AI-powered agentic workflows and systems; the report includes current capabilities, limitations, insights, and ideas for future development of AI agent design.	Paper, Tweet	
9) LM In-Context Recall is Prompt Dependent - analyzes the in-context recall performance of different LLMs using several needle-in-a-haystack tests; shows various LLMs recall facts at different lengths and depths; finds that a model's recall performance is significantly affected by small changes in the prompt; the interplay between prompt content and training data can degrade the response quality; the recall ability of a model can be improved with increasing size, enhancing the attention mechanism, trying different training strategies, and applying fine-tuning.	Paper, Tweet	
10) A Survey on State Space Models - a survey paper on state space models (SSMs) with experimental comparison and analysis; it reviews current SSMs, improvements compared to alternatives, challenges, and their applications.	Paper, Tweet	
Top ML Papers of the Week (April 8 - April 14) - 2024


Paper	Links	
1) Leave No Context Behind - integrates compressive memory into a vanilla dot-product attention layer; the goal is to enable Transformer LLMs to effectively process infinitely long inputs with bounded memory footprint and computation; proposes a new attention technique called Infini-attention which incorporates a compressive memory module into a vanilla attention mechanism; it builds in both masked local attention and long-term linear attention into a single Transformer block; this allows the Infini-Transformer model to efficiently handle both long and short-range contextual dependencies; outperforms baseline models on long-context language modeling with a 114x compression ratio of memory.	Paper, Tweet	
2) OpenEQA - proposes an open-vocabulary benchmark dataset to measure the capabilities of AI models to perform embodied question answering (EQA); it contains 1600 human-generated questions composed from 180 real-world environments; also provides an LLM-powered evaluation protocol for the task and shows that models like GPT-4V are significantly behind human-level performance.	Paper, Tweet	
3) CodeGemma - a family of open code LLMs based on Gemma; CodeGemma 7B models excel in mathematical reasoning and match the code capabilities of other open models; the instruction-tuned CodeGemma 7B model is the more powerful model for Python coding as assessed via the HumanEval benchmark; results also suggest that the model performs best on GSM8K among 7B models; the CodeGemma 2B model achieves SoTA code completion and is designed for fast code infilling and deployment in latency-sensitive settings.	Paper, Tweet	
4) LM-Guided Chain-of-Thought - applies knowledge distillation to a small LM with rationales generated by the large LM with the hope of narrowing the gap in reasoning capabilities; the rationale is generated by the lightweight LM and the answer prediction is then left for the frozen large LM; this resource-efficient approach avoids the need to fine-tune the large model and instead offloads the rationale generation to the small language model; the knowledge-distilled LM is further optimized with reinforcement learning using several rational-oriented and task-oriented reward signals; the LM-guided CoT prompting approach proposed in this paper outperforms both standard prompting and CoT prompting. Self-consistency decoding also enhances performance.	Paper, Tweet	
5) Best Practices and Lessons on Synthetic Data - an overview by Google DeepMind on synthetic data research, covering applications, challenges, and future directions; discusses important topics when working with synthetic data such as ensuring quality, factuality, fidelity, unbiasedness, trustworthiness, privacy, and more.	Paper, Tweet	
6) Reasoning with Intermediate Revision and Search - presents an approach for general reasoning and search on tasks that can be decomposed into components; the proposed graph-based framework, THOUGHTSCULPT, incorporates iterative self-revision capabilities and allows an LLM to build an interwoven network of thoughts; unlike other approaches such as Tree-of-thoughts that shape the reasoning process using a tree, this new approach incorporates Monte Carlo Tree Search (MCTS) to efficiently navigate the search space; due to its ability for continuous thought iteration, THOUGHTSCULPT is particularly suitable for tasks such as open-ended generation, multip-step reasoning, and creative ideation.	Paper, Tweet	
7) Overview of Multilingual LLMs - a survey on multilingual LLMs including a thorough review of methods, a taxonomy, emerging frontiers, challenges, and resources to advance research	Paper, Tweet	
8) The Physics of Language Models - investigates knowledge capacity scaling laws where it evaluates a model’s capability via loss or benchmarks, to estimate the number of knowledge bits a model stores; reports that "Language models can and only can store 2 bits of knowledge per parameter, even when quantized to int8, and such knowledge can be flexibly extracted for downstream applications. Consequently, a 7B model can store 14B bits of knowledge, surpassing the English Wikipedia and textbooks combined based on our estimation."	Paper, Tweet	
9) Aligning LLMs to Quote from Pre-Training Data - proposes techniques to align LLMs to leverage memorized information quotes directly from pre-training data; the alignment approach is not only able to generate high-quality quoted verbatim statements but overall preserve response quality; it leverages a synthetic preference dataset for quoting without any human annotation and aligns the target model to quote using preference optimization.	Paper, Tweet	
10) The Influence Between NLP and Other Fields - aims to quantify the degree of influence between 23 fields of study and NLP; the cross-field engagement of NLP has declined from 0.58 in 1980 to 0.31 in 2022; the study also finds that NLP citations are dominated by CS which accounts for over 80% of citations with emphasis on AI, ML, and information retrieval; overall, NLP is growing more insular -- higher growth of intra-field citation and a decline in multidisciplinary works.	Paper, Tweet	
Top ML Papers of the Week (April 1 - April 7) - 2024


Paper	Links	
1) Many-shot Jailbreaking - proposes a jailbreaking technique called many-shot jailbreaking to evade the safety guardrails of LLMs; this jailbreaking technique exploits the longer context window supported by many modern LLMs; it includes a very large number of faux dialogues (~256) preceding the final question which effectively steers the model to produce harmful responses.	Paper, Tweet	
2) SWE-Agent - a new open-source agentic system that can automatically solve GitHub issues with similar accuracy as Devin on the SWE-bench; the agent interacts with a specialized terminal and enables important processing of files and executable tests to achieve good performance; on SWE-bench, SWE-agent resolves 12.29% of issues, achieving the state-of-the-art performance on the full test set.	Paper, Tweet	
3) Mixture-of-Depths - demonstrates that transformer models can learn to efficiently and dynamically allocate FLOPs to specific positions in a sequence; this helps to optimize the allocation along the sequence for different layers across model depth; findings suggest that for a given FLOP budget models can be trained to perform faster and better than their baseline counterparts.	Paper, Tweet	
4) Local Context LLMs Struggle with Long In-Context Learning - finds that after evaluating 13 long-context LLMs on long in-context learning the LLMs perform relatively well under the token length of 20K. However, after the context window exceeds 20K, most LLMs except GPT-4 will dip dramatically.	Paper, Tweet	
5) Visualization-of-Thought - inspired by a human cognitive capacity to imagine unseen worlds, this new work proposes Visualization-of-Thought (VoT) prompting to elicit spatial reasoning in LLMs; VoT enables LLMs to "visualize" their reasoning traces, creating internal mental images, that help to guide subsequent reasoning steps; when tested on multi-hop spatial reasoning tasks like visual tiling and visual navigation, VoT outperforms existing multimodal LLMs.	Paper, Tweet	
6) The Unreasonable Ineffectiveness of the Deeper Layers - finds that a simple layer-pruning strategy of popular open-weight pretraining LLMs shows minimal performance degradation until after a large fraction (up to half) of the layers are removed; using a layer similarity mechanism optimal blocks are identified and pruned followed by a small amount of fine-tuning to heal damage	Paper, Tweet	
7) JetMoE - an 8B model trained with less than $ 0.1 million cost but outperforms LLaMA2-7B; shows that LLM training can be much cheaper than generally thought; JetMoE-8B has 24 blocks where each block has two MoE layers: Mixture of Attention heads (MoA) and Mixture of MLP Experts (MoE); each MoA and MoE layer has 8 experts, and 2 experts are activated for each input token with 2.2B active parameters.	Paper, Tweet	
8) Representation Finetuning for LMs - proposes a method for representation fine-tuning (ReFT) that operates on a frozen base model and learns task-specific interventions on hidden representations; in other words, by manipulating a small fraction of model representations it is possible to effectively steer model behavior to achieve better downstream performance at inference time; also proposes LoReFT as a drop-in replacement for PEFTs that is 10-50x more parameter efficient.	Paper, Tweet	
9) Advancing LLM Reasoning - proposes a suite of LLMs (Eurus) optimized for reasoning and achieving SoTA among open-source models on tasks such as mathematics and code generation; Eurus-70B outperforms GPT-3.5 Turbo in reasoning largely due to a newly curated, high-quality alignment dataset designed for complex reasoning tasks; the data includes instructions with preference tree consisting of reasoning chains, multi-turn interactions and pairwise data for preference learning.	Paper, Tweet	
10) Training LLMs over Neurally Compressed Text - explores training LLMs with neural text compressors; the proposed compression technique segments text into blocks that each compress to the same bit length; the approach improves at scale and outperforms byte-level baselines on both perplexity and inference speed benchmarks; latency is reduced to the shorter sequence length	Paper, Tweet	
Top ML Papers of the Week (March 26 - March 31) - 2024


Paper	Links	
1) DBRX - a new 132B parameter open LLM that outperforms all the established open-source models on common benchmarks like MMLU and GSM8K; DBRX was pretrained on 12T tokens (text and code) and uses a mixture-of-experts (MoE) architecture; its inference is up to 2x faster than LLaMA2-70B and is about 40% of the size of Grok-1 in terms of both total and active parameter counts; there is also DBRX Instruct which demonstrates good performance in programming and mathematics; while DBRX is trained as a general-purpose LLM, it still surpasses CodeLLaMa-70 Instruct, a model built explicitly for code generation.	Paper, Tweet	
2) Grok-1.5 - xAI’s latest long-context LLM for advanced understanding and reasoning and problem-solving capabilities; Grok-1.5 achieved a 50.6% score on the MATH benchmark and a 90% score on the GSM8K benchmark; this model can process long contexts of up to 128K tokens and demonstrates powerful retrieval capabilities.	Paper, Tweet	
3) SEEDS - a generative AI model based on diffusion models that shows powerful capabilities to quantify uncertainty in weather forecasting; it can generate a large ensemble conditioned on as few as one or two forecasts from an operational numerical weather prediction system.	Paper, Tweet	
4) LLMs for University-Level Coding Course - finds that the latest LLMs have not surpassed human proficiency in physics coding assignments; also finds that GPT-4 significantly outperforms GPT-3.5 and prompt engineering can further enhance performance.	Paper, Tweet	
5) Mini-Gemini - a simple framework to enhance multi-modality vision models; specifically, visual tokens are enhanced through an additional visual encoder for high-resolution refinement without token increase; achieves top performance in several zero-shot benchmarks and even surpasses the developed private models.	Paper, Tweet	
6) Long-form factuality in LLMs - investigates long-form factuality in open-domain by generating a prompt set of questions including 38 topics; also proposes an LLM-based agent to perform evaluation for the task; finds that LLM agents can achieve superhuman rating performance and is reported to be 20 times cheaper than human annotations.	Paper, Tweet	
7) Agent Lumos - a unified framework for training open-source LLM-based agents; it consists of a modular architecture with a planning module that can learn subgoal generation and a module trained to translate them to action with tool usage.	Paper, Tweet	
8) AIOS - an LLM agent operation system that integrates LLMs into operation systems as a brain; the agent can optimize resource allocation, context switching, enable concurrent execution of agents, tool service, and even maintain access control for agents.	Paper, Tweet	
9) FollowIR - a dataset with instruction evaluation benchmark and a separate set for teaching information retrieval model to follow real-world instructions; a FollowIR-7B model has significant improvements (over 13%) after fine-tuning on a training set.	Paper, Tweet	
10) LLM2LLM - an iterative data augmentation strategy that leverages a teacher LLM to enhance a small seed dataset by augmenting additional data that can be used to effectively fine-tune models; it significantly enhances the performance of LLMs in the low-data regime, outperforming both traditional fine-tuning and other data augmentation baselines.	Paper, Tweet	
Top ML Papers of the Week (March 18 - March 25) - 2024


Paper	Links	
1) Grok-1 - a mixture-of-experts model with 314B parameters which includes the open release of the base model weights and network architecture; the MoE model activates 25% of the weights for a given token and its pretraining cutoff date is October 2023.	Paper, Tweet	
2) Evolutionary Model Merge - an approach for automating foundation model development using evolution to combine open-source models; facilitates cross-domain merging where a Japanese Math LLM achieved state-of-the-art performance on Japanese LLM benchmarks, even surpassing models with significantly more parameters, despite not explicitly trained for these tasks.	Paper, Tweet	
3) TacticAI - an AI-powered assistant for football tactics developed and evaluated in collaboration with domain experts from Liverpool FC; the systems offer coaches a way to sample and explore alternative player setups for a corner kick routine and select the tactic with the highest predicted likelihood of success; TacticAI’s model suggestions are favored over existing tactics 90% of the time and it offers an effective corner kick retrieval system.	Paper, Tweet	
4) Tool Use in LLMs - provides an overview of tool use in LLMs, including a formal definition of the tool-use paradigm, scenarios where LLMs leverage tool usage, and for which tasks this approach works well; it also provides an analysis of complex tool usage and summarize testbeds and evaluation metrics across LM tooling works.	Paper, Tweet	
5) Step-by-Step Comparisons Make LLMs Better Reasoners - proposes RankPrompt, a prompting method to enable LLMs to self-rank their responses without additional resources; this self-ranking approach ranks candidates through a systematic, step-by-step comparative evaluation; it seems to work well as it leverages the capabilities of LLMs to generate chains of comparisons as demonstrations; RankPrompt significantly enhances the reasoning performance of ChatGPT and GPT-4 on many arithmetic and commonsense reasoning tasks.	Paper, Tweet	
6) LLM4Decompile - a family of open-access decompilation LLMs ranging from 1B to 33B parameters; these models are trained on 4 billion tokens of C source code and corresponding assembly code; the authors also introduce Decompile-Eval, a dataset for assessing re-compatibility and re-executability for decompilation and evaluating with a perspective of program semantics; LLM4Decompile demonstrates the capability to decompile 21% of the assembly code, achieving a 50% improvement over GPT-4.	Paper, Tweet	
7) Agent-FLAN - designs data and methods to effectively fine-tune language models for agents, referred to as Agent-FLAN; this enables Llama2-7B to outperform prior best works by 3.5% across various agent evaluation datasets; Agent-FLAN greatly alleviates the hallucination issues and consistently improves the agent capability of LLMs when scaling model sizes while generally improving the LLM.	Paper, Tweet	
8) LLMs Leak Proprietary Information - shows that it’s possible to learn a large amount of non-public information about an API-protected LLM using the logits; with a relatively small number of API queries, the approach estimates that the embedding size of OpenAI's gpt-3.5-turbo to be about 4,096; the paper also proposes guardrails against the attacks used	Paper, Tweet	
9) DROID - an open-source, large-scale robot manipulation dataset to train and build more capable and robust robotic manipulation policies; it contains 76K demonstration trajectories, collected across 564 scenes and 86 tasks; training with DROID leads to higher performing policies and generalization.	Paper, Tweet	
10) Retrieval-Augmented Fine-Tuning - combines the benefits of RAG and fine-tuning to improve a model's ability to answer questions in "open-book" in-domain settings; combining it with RAFT's CoT-style response helps to improve reasoning.	Paper, Tweet	
Top ML Papers of the Week (March 11 - March 17) - 2024


Paper	Links	
1) SIMA - a generalist AI agent for 3D virtual environments that follows natural-language instructions in a broad range of 3D virtual environments and video games; SIMA is evaluated across 600 basic skills, spanning navigation, object interaction, and menu use. Language seems to be a huge factor in performance.	Paper, Tweet	
2) Retrieval Augmented Thoughts - shows that iteratively revising a chain of thoughts with information retrieval can significantly improve LLM reasoning and generation in long-horizon generation tasks; the key idea is that each thought step is revised with relevant retrieved information to the task query, the current and past thought steps; Retrieval Augmented Thoughts (RAT) can be applied to different models like GPT-4 and CodeLlama-7B to improve long-horizon generation tasks (e.g., creative writing and embodied task planning); RAT is a zero-shot prompting approach and provides significant improvements to baselines that include zero-shot CoT prompting, vanilla RAG, and other baselines.	Paper, Tweet	
3) LMs Can Teach Themselves to Think Before Speaking - presents a generalization of STaR, called Quiet-STaR, to enable language models (LMs) to learn to reason in more general and scalable ways; Quiet-STaR enables LMs to generate rationales at each token to explain future text; it proposes a token-wise parallel sampling algorithm that helps improve LM predictions by efficiently generating internal thoughts; the rationale generation is improved using REINFORCE.	Paper, Tweet	
4) Knowledge Conflicts for LLMs - an overview of the common issue of knowledge conflict when working with LLMs; the survey paper categorizes these conflicts into context-memory, inter-context, and intra-memory conflict; it also provides insights into causes and potential ways to mitigate these knowledge conflict issues.	Paper, Tweet	
5) Stealing Part of a Production Language Model - presents the first model-stealing attack that extracts information from production language models like ChatGPT or PaLM-2; shows that it's possible to recover the embedding projection layer of a transformer-based model through typical API access; as an example, the entire projection matrix was extracted from the OpenAI ada and babbage models for under $20.	Paper, Tweet	
6) Branch-Train-MiX - proposes mixing expert LLMs into a Mixture-of-Experts LLM as a more compute-efficient approach for training LLMs; it's shown to be more efficient than training a larger generalist LLM or several separate specialized LLMs; the approach, BTX, first trains (in parallel) multiple copies of a seed LLM specialized in different domains (i.e., expert LLMs) and merges them into a single LLM using MoE feed-forward layers, followed by fine-tuning of the overall unified model.	Paper, Tweet	
7) LLMs Predict Neuroscience Results - proposes a benchmark, BrainBench, for evaluating the ability of LLMs to predict neuroscience results; finds that LLMs surpass experts in predicting experimental outcomes; an LLM tuned on neuroscience literature was shown to perform even better.	Paper, Tweet	
8) C4AI Command-R - a 35B parameter model, with a context length of 128K, optimized for use cases that include reasoning, summarization, and question answering; Command-R has the capability for multilingual generation evaluated in 10 languages and performant tool use and RAG capabilities; it has been released for research purposes.	Paper, Tweet	
9) Is Cosine-Similarity Really About Simirity? - studies embeddings derived from regularized linear models and derive analytically how cosine-similarity can yield arbitrary and meaningless similarities; also finds that for some linear models, the similarities are not even unique and others are controlled by regularization; the authors caution against blindly using cosine similarity and presents considerations and alternatives.	Paper, Tweet	
10) Multimodal LLM Pre-training - provides a comprehensive overview of methods, analysis, and insights into multimodal LLM pre-training; studies different architecture components and finds that carefully mixing image-caption, interleaved image-text, and text-only data is key for state-of-the-art performance; it also proposes a family of multimodal models up to 30B parameters that achieve SOTA in pre-training metrics and include properties such as enhanced in-context learning, multi-image reasoning, enabling few-shot chain-of-thought prompting.	Paper, Tweet	
Top ML Papers of the Week (March 4 - March 10) - 2024


Paper	Links	
1) Claude 3 - consists of a family of three models (Claude 3 Haiku, Claude 3 Sonnet, and Claude 3 Opus); Claude 3 Opus (the strongest model) seems to outperform GPT-4 on common benchmarks like MMLU and HumanEval; Claude 3 capabilities include analysis, forecasting, content creation, code generation, and converting in non-English languages like Spanish, Japanese, and French; 200K context windows supported but can be extended to 1M token to select customers; the models also have strong vision capabilities for processing formats like photos, charts, and graphs; Anthropic claims these models have a more nuanced understanding of requests and make fewer refusals.	Paper, Tweet	
2) Robust Evaluation of Reasoning - proposes functional benchmarks for the evaluation of the reasoning capabilities of LLMs; finds that there is a reasoning gap with current models from 58.35% to 80.31%; however, the authors also report that those gaps can be reduced with more sophisticated prompting strategies.	Paper, Tweet	
3) GaLore - proposes a memory-efficient approach for training LLM through low-rank projection; the training strategy allows full-parameter learning and is more memory-efficient than common low-rank adaptation methods such as LoRA; reduces memory usage by up to 65.5% in optimizer states while maintaining both efficiency and performance for pre-training on LLaMA 1B and 7B architectures.	Paper, Tweet	
4) Can LLMs Reason and Plan? - a new position paper discusses the topic of reasoning and planning for LLMs; here is a summary of the author's conclusion: "To summarize, nothing that I have read, verified, or done gives me any compelling reason to believe that LLMs do reasoning/planning, as normally understood. What they do instead, armed with web-scale training, is a form of universal approximate retrieval, which, as I have argued, can sometimes be mistaken for reasoning capabilities".	Paper, Tweet	
5) RAG for AI-Generated Content - provides an overview of RAG used in different generation scenarios like code, image, and audio, including a taxonomy of RAG enhancements with reference to key papers.	Paper, Tweet	
6) KnowAgent - proposes an approach to enhance the planning capabilities of LLMs through explicit action knowledge; uses an action knowledge base and a knowledgeable self-learning phase to guide the model's action generation, mitigate planning hallucination, and enable continuous improvement; outperforms existing baselines and shows the potential of integrating external action knowledge to streamline planning with LLMs and solve complex planning challenges.	Paper, Tweet	
7) Sora Overview - a comprehensive review of Sora and some of the key developments powering this model, including limitations and opportunities of large vision models.	Paper, Tweet	
8) LLM for Law - introduces SaulLM-7B, a large language model for the legal domain explicitly designed for legal text comprehension and generation; presents an instructional fine-tuning method that leverages legal datasets to further enhance performance in legal tasks.	Paper, Tweet	
9) Design2Code - investigates the use of multimodal LLMs for converting a visual design into code implementation which is key for automating front-end engineering; introduces a benchmark of 484 diverse real-world webpages and a set of evaluation metrics to measure the design-to-code capability; further develops a suite of multimodal prompting methods and show their effectiveness on GPT-4V and Gemini Pro Vision; an open-source fine-tuned Design2Code matches the performance of Gemini Pro Vision, however, GPT-4V performs the best on the task.	Paper, Tweet	
10) TripoSR - a transformer-based 3D reconstruction model for fast feed-forward 3D generation; it can produce 3D mesh from a single image in under 0.5 seconds; improvement includes better data processing, model design, and training.	Paper, Tweet	
Top ML Papers of the Week (February 26 - March 3) - 2024


Paper	Links	
1) Genie - a foundation model trained from internet videos and with the ability to generate a variety of action-controllable 2D worlds given an image prompt; Genie has 11B parameters and consists of a spatiotemporal video tokenizer, an autoregressive dynamic model, and a scalable latent action model; the latent action space enables training agents to imitate behaviors from unseen video which is promising for building more generalist agents.	Paper, Tweet	
2) Mistral Large - a new LLM with strong multilingual, reasoning, maths, and code generation capabilities; features include: 1) 32K tokens context window, 2) native multilingual capacities, 3) strong abilities in reasoning, knowledge, maths, and coding benchmarks, and 4) function calling and JSON format natively supported.	Paper, Tweet	
3) The Era of 1-bit LLMs - introduces a high-performing and cost-effective 1-bit LLM variant called BitNet b1.58 where every parameter is a ternary {-1, 0, 1}; given the same model size and training tokens, BitNet b1.58 can match the perplexity and task performance of a full precision Transformer LLM (i.e., FP16); the benefits of this 1-bit LLM are significantly better latency, memory, throughout, and energy consumption.	Paper, Tweet	
4) Dataset for LLMs - a comprehensive overview (180+ pages) and analysis of LLM datasets.	Paper, Tweet	
5) LearnAct - explores open-action learning for language agents through an iterative learning strategy that creates and improves actions using Python functions; on each iteration, the proposed framework (LearnAct) expands the action space and enhances action effectiveness by revising and updating available actions based on execution feedback; the LearnAct framework was tested on Robotic planning and AlfWorld environments; it improves agent performance by 32% in AlfWorld compared to ReAct+Reflexion.	Paper, Tweet	
6) EMO - a new framework for generating expressive video by utilizing a direct audio-to-video synthesis approach; by leveraging an Audio2Video diffusion model it bypasses the need for intermediate 3D models or facial landmarks; EMO can produce convincing speaking videos and singing videos in various styles while outperforming existing methods in terms of expressiveness and realism.	Paper, Tweet	
7) On the Societal Impact of Open Foundation Models - a position paper with a focus on open foundation models and their impact, benefits, and risks; proposes a risk assessment framework for analyzing risk and explains why the marginal risk of open foundation models is low in some cases; it also offers a more grounded assessment of the societal impact of open foundation models.	Paper, Tweet	
8) StarCoder 2 - a family of open LLMs for code with three different sizes (3B, 7B, and 15B); the 15B model was trained on 14 trillion tokens and 600+ programming languages with a context window of 16K token and employing a fill-in-the-middle objective; it matches 33B+ models on many evaluation like code completion, code reasoning, and math reasoning aided through PAL.	Paper, Tweet	
9) LLMs on Tabular Data - an overview of LLMs for tabular data tasks including key techniques, metrics, datasets, models, and optimization approaches; it covers limitations and unexplored ideas with insights for future research directions.	Paper, Tweet	
10) PlanGPT - shows how to leverage LLMs and combine multiple approaches like retrieval augmentation, fine-tuning, tool usage, and more; the proposed framework is applied to urban and spatial planning but there are a lot of insights and practical tips that apply to other domains.	Paper, Tweet	
Top ML Papers of the Week (February 19 - February 25) - 2024


Paper	Links	
1) Stable Diffusion 3 - a suite of image generation models ranging from 800M to 8B parameters; combines diffusion transformer architecture and flow matching for improved performance in multi-subject prompts, image quality, and spelling abilities; technical report to be published soon and linked here.	Paper, Tweet	
2) Gemma - a series of open models inspired by the same research and tech used for Gemini; includes 2B (trained on 2T tokens) and 7B (trained on 6T tokens) models including base and instruction-tuned versions; trained on a context length of 8192 tokens; generally outperforms Llama 2 7B and Mistral 7B.	Paper, Tweet	
3) LLMs for Data Annotation - an overview and a good list of references that apply LLMs for data annotation; includes a taxonomy of methods that employ LLMs for data annotation; covers three aspects: LLM-based data annotation, assessing LLM-generated annotations, and learning with LLM-generated annotations.	Paper, Tweet	
4) GRIT - presents generative representational instruction tuning where an LLM is trained to perform both generative and embedding tasks and designed to distinguish between them via the instructions; produces new state-of-the-art on MTEB and the unification is reported to speed up RAG by 60% for long documents.	Paper, Tweet	
5) LoRA+ - proposes LoRA+ which improves performance and finetuning speed (up to ∼ 2X speed up), at the same computational cost as LoRA; the key difference between LoRA and LoRA+ is how the learning rate is set; LoRA+ sets different learning rates for LoRA adapter matrices while in LoRA the learning rate is the same.	Paper, Tweet	
6) Revisiting REINFORCE in RLHF - shows that many components of PPO are unnecessary in an RLHF context; it also shows that a simpler REINFORCE variant outperforms both PPO and newly proposed alternatives such as DPO and RAFT; overall, it shows that online RL optimization can be beneficial and low cost.	Paper, Tweet	
7) Recurrent Memory Finds What LLMs Miss - explores the capability of transformer-based models in extremely long context processing; finds that both GPT-4 and RAG performance heavily rely on the first 25% of the input, which means there is room for improved context processing mechanisms; reports that recurrent memory augmentation of transformer models achieves superior performance on documents of up to 10 million tokens.	Paper, Tweet	
8) When is Tree Search Useful for LLM Planning - investigates how LLM solves multi-step problems through a framework consisting of a generator, discriminator, and planning method (e.g., iterative correction and tree search); reports that planning methods demand discriminators with at least 90% accuracy but current LLMs don’t demonstrate these discrimination capabilities; finds that tree search is at least 10 to 20 times slower but regardless of it good performance it’s impractical for real-world applications.	Paper, Tweet	
9) CoT Reasoning without Prompting - proposes a chain-of-thought (CoT) decoding method to elicit the reasoning capabilities from pre-trained LLMs without explicit prompting; claims to significantly enhance a model’s reasoning capabilities over greedy decoding across reasoning benchmarks; finds that the model's confidence in its final answer increases when CoT is present in its decoding path.	Paper, Tweet	
10) OpenCodeInterpreter - a family of open-source systems for generating, executing, and iteratively refining code; proposes a dataset of 68K multi-turn interactions; integrates execution and human feedback for dynamic code refinement and produces high performance on benchmarks like HumalEval and EvalPlus.	Paper, Tweet	
Top ML Papers of the Week (February 12 - February 18) - 2024


Paper	Links	
1) Sora - a text-to-video AI model that can create videos of up to a minute of realistic and imaginative scenes given text instructions; it can generate complex scenes with multiple characters, different motion types, and backgrounds, and understand how they relate to each other; other capabilities include creating multiple shots within a single video with persistence across characters and visual style.	Paper, Tweet	
2) Gemini 1.5 - a compute-efficient multimodal mixture-of-experts model that focuses on capabilities such as recalling and reasoning over long-form content; it can reason over long documents potentially containing millions of tokens, including hours of video and audio; improves the state-of-the-art performance in long-document QA, long-video QA, and long-context ASR. Gemini 1.5 Pro matches or outperforms Gemini 1.0 Ultra across standard benchmarks and achieves near-perfect retrieval (>99%) up to at least 10 million tokens, a significant advancement compared to other long-context LLMs.	Paper, Tweet	
3) V-JEPA - a collection of vision models trained on a feature prediction objective using 2 million videos; relies on self-supervised learning and doesn’t use pretrained image encoders, text, negative examples, reconstruction, or other supervision sources; claims to achieve versatile visual representations that perform well on both motion and appearance-based tasks, without adaption of the model’s parameters.	Paper, Tweet	
4) Large World Model - a general-purpose 1M context multimodal model trained on long videos and books using RingAttention; sets new benchmarks in difficult retrieval tasks and long video understanding; uses masked sequence packing for mixing different sequence lengths, loss weighting, and model-generated QA dataset for long sequence chat; open-sources a family of 7B parameter models that can process long text and videos of over 1M tokens.	Paper, Tweet	
5) The boundary of neural network trainability is fractal - finds that the boundary between trainable and untrainable neural network hyperparameter configurations is fractal; observes fractal hyperparameter landscapes for every neural network configuration and deep linear networks; also observes that the best-performing hyperparameters are at the end of stability.	Paper, Tweet	
6) OS-Copilot - a framework to build generalist computer agents that interface with key elements of an operating system like Linux or MacOS; it also proposes a self-improving embodied agent for automating general computer tasks; this agent outperforms the previous methods by 35% on the general AI assistants (GAIA) benchmark.	Paper, Tweet	
7) TestGen-LLM - uses LLMs to automatically improve existing human-written tests; reports that after an evaluation on Reels and Stories products for Instagram, 75% of TestGen-LLM's test cases were built correctly, 57% passed reliably, and 25% increased coverage.	Paper, Tweet	
8) ChemLLM - a dedicated LLM trained for chemistry-related tasks; claims to outperform GPT-3.5 on principal tasks such as name conversion, molecular caption, and reaction prediction; it also surpasses GPT-4 on two of these tasks.	Paper, Tweet	
9) Survey of LLMs - reviews three popular families of LLMs (GPT, Llama, PaLM), their characteristics, contributions, and limitations; includes a summary of capabilities and techniques developed to build and augment LLM; it also discusses popular datasets for LLM training, fine-tuning, and evaluation, and LLM evaluation metrics; concludes with open challenges and future research directions.	Paper, Tweet	
10) LLM Agents can Hack - shows that LLM agents can automatically hack websites and perform tasks like SQL injections without human feedback or explicit knowledge about the vulnerability beforehand; this is enabled by an LLM’s tool usage and long context capabilities; shows that GPT-4 is capable of such hacks, including finding vulnerabilities in websites in the wild; open-source models did not show the same capabilities.	Paper, Tweet	
Top ML Papers of the Week (February 5 - February 11) - 2024


Paper	Links	
1) Grandmaster-Level Chess Without Search - trains a 270M parameter transformer model with supervised learning on a dataset of 10 million chess games with up to 15 billion data points; reaches a Lichess blitz Elo of 2895 against humans, and solves a series of challenging chess puzzles; it shows the potential of training at scale for chess and without the need for any domain-specific tweaks or explicit search algorithms.	Paper, Tweet	
2) AnyTool - an LLM-based agent that can utilize 16K APIs from Rapid API; proposes a simple framework consisting of 1) a hierarchical API-retriever to identify relevant API candidates to a query, 2) a solver to resolve user queries, and 3) a self-reflection mechanism to reactivate AnyTool if the initial solution is impracticable; this tool leverages the function calling capability of GPT-4 so no further training is needed; the hierarchical API-retriever is inspired by a divide-and-conquer approach to help reduce the search scope of the agents which leads to overcoming limitations around context length in LLMs; the self-reflection component helps with resolving easy and complex queries efficiently.	Paper, Tweet	
3) A Phase Transition between Positional and Semantic Learning in a Solvable Model of Dot-Product Attention - investigates and expands the theoretical understanding of learning with attention layers by exploring the interplay between positional and semantic attention; it employs a toy model of dot-product attention and identifies an emergent phase transition between semantic and positional learning; shows that if provided with sufficient data, dot-product attention layer outperforms a linear positional baseline when using the semantic mechanism.	Paper, Tweet	
4) Indirect Reasoning with LLMs - proposes an indirect reasoning method to strengthen the reasoning power of LLMs; it employs the logic of contrapositives and contradictions to tackle IR tasks such as factual reasoning and mathematic proof; it consists of two key steps: 1) enhance the comprehensibility of LLMs by augmenting data and rules (i.e., the logical equivalence of contrapositive), and 2) design prompt templates to stimulate LLMs to implement indirect reasoning based on proof by contradiction; experiments on LLMs like GPT-3.5-turbo and Gemini Pro show that the proposed method enhances the overall accuracy of factual reasoning by 27.33% and mathematic proof by 31.43% compared to traditional direct reasoning methods.	Paper, Tweet	
5) ALOHA 2 - a low-cost system for bimanual teleoperation that improves the performance, user-friendliness, and durability of ALOHA; efforts include hardware improvements such as grippers and gravity compensation with a higher quality simulation model; this potentially enables large-scale data collection on more complex tasks to help advanced research in robot learning.	Paper, Tweet	
6) More Agents is All You Need - presents a study on the scaling property of raw agents instantiated by LLMs; finds that performance scales when increasing agents by simply using a sampling-and-voting method.	Paper, Tweet	
7) Self-Discovered Reasoning Structures - proposes a new framework, Self-Discover, that enables LLMs to select from multiple reasoning techniques (e.g., critical thinking and thinking step-by-step) to compose task-specific reasoning strategies; outperforms CoT (applied to GPT-4 and PaLM 2) on BigBench-Hard experiments and requires 10-40x fewer inference compute than other inference-intensive methods such as CoT-Self-Consistency; the self-discovered reasoning structures are also reported to transfer well between LLMs and small language models (SLMs).	Paper, Tweet	
8) DeepSeekMath - continues pretraining a code base model with 120B math-related tokens; introduces GRPO (a variant to PPO) to enhance mathematical reasoning and reduce training resources via a memory usage optimization scheme; DeepSeekMath 7B achieves 51.7% on MATH which approaches the performance level of Gemini-Ultra (53.2%) and GPT-4 (52.9%); when self-consistency is used the performance improves to 60.9%.	Paper, Tweet	
9) LLMs for Table Processing - provides an overview of LLMs for table processing, including methods, benchmarks, prompting techniques, and much more.	Paper, Tweet	
10) LLM-based Multi-Agents - discusses the essential aspects of LLM-based multi-agent systems; it includes a summary of recent applications for problem-solving and word simulation; it also discusses datasets, benchmarks, challenges, and future opportunities to encourage further research and development from researchers and practitioners.	Paper, Tweet	
Top ML Papers of the Week (January 29 - February 4) - 2024


Paper	Links	
1) OLMo - introduces Open Language Model (OLMo), a 7B parameter model; it includes open training code, open data, full model weights, evaluation code, and fine-tuning code; it shows strong performance on many generative tasks; there is also a smaller version of it, OLMo 1B.	Paper, Tweet	
2) Advances in Multimodal LLMs - a comprehensive survey outlining design formulations for model architecture and training pipeline around multimodal large language models.	Paper, Tweet	
3) Corrective RAG - proposes Corrective Retrieval Augmented Generation (CRAG) to improve the robustness of generation in a RAG system; the core idea is to implement a self-correct component for the retriever and improve the utilization of retrieved documents for augmenting generation; the retrieval evaluator helps to assess the overall quality of retrieved documents given a query; using web search and optimized knowledge utilization operations can improve automatic self-correction and efficient utilization of retrieved documents.	Paper, Tweet	
4) LLMs for Mathematical Reasoning - introduces an overview of research developments in LLMs for mathematical reasoning; discusses advancements, capabilities, limitations, and applications to inspire ongoing research on LLMs for Mathematics.	Paper, Tweet	
5) Compression Algorithms for LLMs - covers compression algorithms like pruning, quantization, knowledge distillation, low-rank approximation, parameter sharing, and efficient architecture design.	Paper, Tweet	
6) MoE-LLaVA - employs Mixture of Experts tuning for Large Vision-Language Models which constructs a sparse model with a substantial reduction in parameters with a constant computational cost; this approach also helps to address performance degradation associated with multi-modal learning and model sparsity.	Paper, Tweet	
7) Rephrasing the Web - uses an off-the-shelf instruction-tuned model prompted to paraphrase web documents in specific styles and formats such as “like Wikipedia” or “question-answer format” to jointly pre-train LLMs on real and synthetic rephrases; it speeds up pre-training by ~3x, improves perplexity, and improves zero-shot question answering accuracy on many tasks.	Paper, Tweet	
8) Redefining Retrieval in RAG - a study that focuses on the components needed to improve the retrieval component of a RAG system; confirms that the position of relevant information should be placed near the query, the model will struggle to attend to the information if this is not the case; surprisingly, it finds that related documents don't necessarily lead to improved performance for the RAG system; even more unexpectedly, irrelevant and noisy documents can help drive up accuracy if placed correctly.	Paper, Tweet	
9) Hallucination in LVLMs - discusses hallucination issues and techniques to mitigate hallucination in Large Vision-Language Models (LVLM); it introduces LVLM hallucination evaluation methods and benchmarks; provides tips and a good analysis of the causes of LVLM hallucinations and potential ways to mitigate them.	Paper, Tweet	
10) SliceGPT - a new LLM compression technique that proposes a post-training sparsification scheme that replaces each weight matrix with a smaller dense matrix; helps reduce the embedding dimension of the network and can remove up to 20% of model parameters for Llama2-70B and Phi-2 models while retaining most of the zero-shot performance of the dense models.	Paper, Tweet	
Top ML Papers of the Week (January 22 - January 28) - 2024


Paper	Links	
1) Depth Anything - a robust monocular depth estimation solution that can deal with any images under any circumstance; automatically annotates large-scale unlabeled data (~62M) which helps to reduce generalization error; proposes effective strategies to leverage the power of the large-scale unlabeled data; besides generalization ability, it established new state-of-the-art through fine-tuning and even results in an enhanced depth-conditioned ControlNet.	Paper, Tweet	
2) Knowledge Fusion of LLMs - proposes FuseLLM with the core idea of externalizing knowledge from multiple LLMs and transferring their capabilities to a target LLM; leverages the generative distributions of source LLMs to externalize both their collective knowledge and individual strengths and transfer them to the target LLM through continual training; finds that the FuseLLM can improve the performance of the target model across a range of capabilities such as reasoning, common sense, and code generation.	Paper, Tweet	
3) MambaByte - adapts Mamba SSM to learn directly from raw bytes; bytes lead to longer sequences which autoregressive Transformers will scale poorly on; this work reports huge benefits related to faster inference and even outperforms subword Transformers.	Paper, Tweet	
4) Diffuse to Choose - a diffusion-based image-conditioned inpainting model to balance fast inference with high-fidelity while enabling accurate semantic manipulations in a given scene content; outperforms existing zero-shot diffusion inpainting methods and even few-shot diffusion personalization algorithms such as DreamPaint.	Paper, Tweet	
5) WARM - introduces weighted averaged rewards models (WARM) that involve fine-tuning multiple rewards models and then averaging them in the weight space; average weighting improves efficiency compared to traditional prediction ensembling; it improves the quality and alignment of LLM predictions.	Paper, Tweet	
6) Resource-efficient LLMs & Multimodal Models - a survey of resource-efficient LLMs and multimodal foundations models; provides a comprehensive analysis and insights into ML efficiency research, including architectures, algorithms, and practical system designs and implementations.	Paper, Tweet	
7) Red Teaming Visual Language Models - first presents a red teaming dataset of 10 subtasks (e.g., image misleading, multi-modal jailbreaking, face fairness, etc); finds that 10 prominent open-sourced VLMs struggle with the red teaming in different degrees and have up to 31% performance gap with GPT-4V; also applies red teaming alignment to LLaVA-v1.5 with SFT using the proposed red teaming dataset, which improves model performance by 10% in the test set.	Paper, Tweet	
8) Lumiere - a text-to-video space-time diffusion model for synthesizing videos with realistic and coherent motion; introduces a Space-Time U-Net architecture to generate the entire temporal duration of a video at once via a single pass; achieves state-of-the-art text-to-video generation results and supports a wide range of content creation tasks and video editing applications, including image-to-video, video inpainting, and stylized generation.	Paper, Tweet	
9) Medusa - a simple framework for LLM inference acceleration using multiple decoding heads that predict multiple subsequent tokens in parallel; parallelization substantially reduces the number of decoding steps; it can achieve over 2.2x speedup without compromising generation quality, while Medusa-2 further improves the speedup to 2.3-3.6x.	Paper, Tweet	
10) AgentBoard - a comprehensive benchmark with an open-source evaluation framework to perform analytical evaluation of LLM agents; helps to assess the capabilities and limitations of LLM agents and demystifies agent behaviors which leads to building stronger and robust LLM agents.	Paper, Tweet	
Top ML Papers of the Week (January 15 - January 21) - 2024


Paper	Links	
1) AlphaGeometry - an AI system that acts as a theorem prover that can solve Olympiad geometry problems without human demonstrations; this system is trained on synthetic data involving millions of theorems and proofs across different levels of complexity; the data is used to train a neural language model that can solve olympiad-level problems and approaches the performance of an average International Mathematical Olympiad (IMO) gold medallist.	Paper, Tweet	
2) AlphaCodium - a code-oriented iterative flow that improves LLMs on code generation; it involves two key steps to improve code generation capabilities in LLMs: i) additional generated data (problem self-reflection and test reasoning) to aid the iterative process, and ii) enriching public tests using additional AI-generated tests; using the CodeContests validation dataset, GPT-4 pass@5 accuracy increased from 19% using a single well-crafted prompt to 44% using the AlphaCodium flow; it even outperforms AlphaCode using a significantly smaller computation budget and 4 orders of magnitude fewer LLM calls.	Paper, Tweet	
3) RAG vs. Finetuning - report discussing the tradeoff between RAG and fine-tuning when using LLMs like Llama 2 and GPT-4; performs a detailed analysis and highlights insights when applying the pipelines on an agricultural dataset; observes that there is an accuracy increase of over 6 p.p. when fine-tuning the model and this is cumulative with RAG, which increases accuracy by 5 p.p. further.	Paper, Tweet	
4) Self-Rewarding Models - proposes a self-alignment method that uses the model itself for LLM-as-a-Judge prompting to provide its rewards during training; Iterative DPO is used for instruction following training using the preference pairs built from the generated data which comes from a self-instruction creation phase; using this approach, fine-tuning a Llama 2 70B model on three iterations can lead to a model that outperforms LLMs like Claude 2 and Gemini Pro on the AlpacaEval 2.0 leaderboard.	Paper, Tweet	
5) Tuning Language Models by Proxy - introduces proxy-tuning, a decoding-time algorithm that modifies logits of a target LLM with the logits’ difference between a small base model and a fine-tuned base model; this can enable a larger target base model to perform as well as would a fine-tuned version of it; proxy-tuning is applied to Llama2-70B using proxies of only 7B size to close 88% of the gap between Llama2-70B and its tuned chat version.	Paper, Tweet	
6) Reasoning with Reinforced Fine-Tuning - proposes an approach, ReFT, to enhance the generalizability of LLMs for reasoning; it starts with applying SFT and then applies online RL for further refinement while automatically sampling reasoning paths to learn from; this differs from RLHF in that it doesn’t utilize a reward model learned from human-labeled data; ReFT demonstrates improved performance and generalization abilities on math problem-solving.	Paper, Tweet	
7) Overview of LLMs for Evaluation - thoroughly surveys the methodologies and explores their strengths and limitations; provides a taxonomy of different approaches involving prompt engineering or calibrating open-source LLMs for evaluation	Paper, Tweet	
8) Patchscopes - proposes a framework that leverages a model itself to explain its internal representations; it decodes information from LLM hidden representations which is possible by “patching” representations into a separate inference pass that encourages the extraction of that information; it can be used to answer questions about an LLM’s computation and can even be used to fix latent multi-hop reasoning errors.	Paper, Tweet	
9) The Unreasonable Effectiveness of Easy Training Data for Hard Tasks - suggests that language models often generalize well from easy to hard data, i.e., easy-to-hard generalization; it argues that it can be better to train on easy data as opposed to hard data, even when the emphasis is on improving performance on hard data, and suggests that the scalable oversight problem may be easier than previously thought.	Paper, Tweet	
10) MoE-Mamba - an approach to efficiently scale LLMs by combining state space models (SSMs) with Mixture of Experts (MoE); MoE-Mamba, outperforms both Mamba and Transformer-MoE; it reaches the same performance as Mamba in 2.2x less training steps while preserving the inference performance gains of Mamba against the Transformer.	Paper, Tweet	
Top ML Papers of the Week (January 8 - January 14) - 2024


Paper	Links	
1) InseRF - a method for text-driven generative object insertion in the Neural 3D scenes; it enables users to provide textual descriptions and a 2D bounding box in a reference viewpoint to generate new objects in 3D scenes; InseRF is also capable of controllable and 3D-consistent object insertion without requiring explicit 3D information as input.	Paper, Tweet	
2) Sleeper Agents - shows that LLMs can learn deceptive behavior that persists through safety training; for instance, an LLM was trained to write secure code for a specified year but given another year can enable exploitable code; this backdoor behavior can persist even when training LLMs with techniques like reinforcement learning and adversarial training.	Paper, Tweet	
3) Blending Is All You Need - shows that effectively combining existing small models of different sizes (6B/13B parameters) can result in systems that can compete with ChatGPT level performance; the goal is to build a collaborative conversational system that can effectively leverage these models to improve engagement and quality of chat AIs and generate more diverse responses.	Paper, Tweet	
4) MagicVideo-V2 - proposes an end-to-end video generation pipeline that integrates the text-to-image model, video motion generator, reference image embedding module, and frame interpolation module; it can generate high-resolution video with advanced fidelity and smoothness compared to other leading and popular text-to-video systems.	Paper, Tweet	
5) Trustworthiness in LLMs - a comprehensive study (100+ pages) of trustworthiness in LLMs, discussing challenges, benchmarks, evaluation, analysis of approaches, and future directions; proposes a set of principles for trustworthy LLMs that span 8 dimensions, including a benchmark across 6 dimensions (truthfulness, safety, fairness, robustness, privacy, and machine ethics); it also presents a study evaluating 16 mainstream LLMs in TrustLLM, consisting of over 30 datasets; while proprietary LLMs generally outperform most open-source counterparts in terms of trustworthiness, there are a few open-source models that are closing the gap.	Paper, Tweet	
6) Prompting LLMs for Table Understanding - a new framework, inspired by Chain-of-Thought prompting, to instruct LLMs to dynamically plan a chain of operations that transforms a complex table to reliably answer the input question; an LLM is used to iteratively generate operations, step-by-step, that will perform necessary transformations to the table (e.g., adding columns or deleting info).	Paper, Tweet	
7) Jailbreaking Aligned LLMs - proposes 40 persuasion techniques to systematically jailbreak LLMs; their adversarial prompts (also referred to as persuasive adversarial prompts) achieve a 92% attack success rate on aligned LLMs, like Llama 2-7B and GPT-4, without specialized optimization.	Paper, Tweet	
8) From LLM to Conversational Agents - proposes RAISE, an advanced architecture to enhance LLMs for conversational agents; it's inspired by the ReAct framework and integrates a dual-component memory system; it utilizes a scratchpad and retrieved examples to augment the agent's capabilities; the scratchpad serves as transient storage (akin to short-term memory) and the retrieval module operates as the agent's long-term memory; this system mirrors human short-term and long-term memory and helps to maintain context and continuity which are key in conversational systems.	Paper, Tweet	
9) Quantifying LLM’s Sensitivity to Spurious Features in Prompt Design - finds that widely used open-source LLMs are extremely sensitive to prompt formatting in few-shot settings; subtle changes in prompt formatting using a Llama 2 13B model can result in a performance difference of up to 76 accuracy points.	Paper, Tweet	
10) Adversarial Machine Learning - a comprehensive survey that covers the current state of adversarial ML with a proper taxonomy of concepts, discussions, adversarial methods, mitigation tactics, and remaining challenges.	Paper, Tweet	
Top ML Papers of the Week (January 1 - January 7) - 2024


Paper	Links	
1) Mobile ALOHA - proposes a system that learns bimanual mobile manipulation with low-cost whole-body teleoperation; it first collects high-quality demonstrations and then performs supervised behavior cloning; finds that co-training with existing ALOHA datasets increases performance on complex mobile manipulation tasks such as sauteing and serving a piece of shrimp, opening a two-door wall cabinet to store heavy cooking pots while keeping the budget under $32K	Paper, Tweet	
2) Mitigating Hallucination in LLMs - summarizes 32 techniques to mitigate hallucination in LLMs; introduces a taxonomy categorizing methods like RAG, Knowledge Retrieval, CoVe, and more; provides tips on how to apply these methods and highlights the challenges and limitations inherent in them.	Paper, Tweet	
3) Self-Play Fine-tuning - shows that without acquiring additional human-annotated data, a supervised fine-tuned LLM can be improved; inspired by self-play, it first uses the LLM to generate its training data from its previous iterations; it then refines its policy by distinguishing the self-generated responses from those obtained from human-annotated data; shows that the method can improve LLM’s performance and outperform models trained via DPO with GPT-4 preference data.	Paper, Tweet	
4) LLaMA Pro - proposes a post-pretraining method to improve an LLM’s knowledge without catastrophic forgetting; it achieves this by tuning expanded identity blocks using only new corpus while freezing the inherited blocks; uses math and code data to train a LLaMA Pro-8.3B initialized from Llama2-7B; these models achieve advanced performance on various benchmarks compared to base models while preserving the original general capabilities.	Paper, Tweet	
5) LLM Augmented LLMs - explore composing existing foundation models with specific models to expand capabilities; introduce cross-attention between models to compose representations that enable new capabilities; as an example, a PaLM2-S model was augmented with a smaller model trained on low-resource languages to improve English translation and arithmetic reasoning for low-resource languages; this was also done with a code-specific model which led to a 40% improvement over the base code model on code generation and explanation tasks.	Paper, Tweet	
6) Fast Inference of Mixture-of-Experts - achieves efficient inference of Mixtral-8x7B models through offloading; it applies separate quantization for attention layers and experts to fit the model in combined GPU and CPU memory; designs a MoE-specific offloading strategy that enables running Mixtral-8x7B on desktop hardware and free-tier Google Colab instances	Paper, Tweet	
7) GPT-4V is a Generalist Web Agent - explores the potential of GPT-4V as a generalist web agent; in particular, can such a model follow natural language instructions to complete tasks on a website? the authors first developed a tool to enable web agents to run on live websites; findings suggest that GPT-4V can complete 50% of tasks on live websites, possible through manual grounding of its textual plans into actions on the websites.	Paper, Tweet	
8) DocLLM - a lightweight extension to traditional LLMs for reasoning over visual documents; focuses on using bounding box information to incorporate spatial layout structure; proposes a pre-training objective that addresses irregular layout and heterogeneous content present in visual documents; it’s then fine-tuned on an instruction-dataset and demonstrate SoTA performance on 14 out of 16 datasets across several document intelligence tasks.	Paper, Tweet	
9) How Code Empowers LLMs - a comprehensive overview of the benefits of training LLMs with code-specific data. Some capabilities include enhanced code generation, enabling reasoning, function calling, automated self-improvements, and serving intelligent agents.	Paper, Tweet	
10) Instruct-Imagen - proposes an image generation model that tackles heterogeneous image generation tasks and generalizes across unseen tasks; it first enhances the model’s ability to ground its generation on external multimodal context and then fine-tunes on image generation tasks with multimodal instructions	Paper, Tweet	


Top ML Papers of the Week (December 25 - December 31)


Paper	Links	
1) CogAgent - presents an 18 billion parameter visual language model specializing in GUI understanding and navigation; supports high-resolution inputs (1120x1120) and shows abilities in tasks such as visual Q&A, visual grounding, and GUI Agent; achieves state of the art on 5 text-rich and 4 general VQA benchmarks.	Paper, Tweet	
2) From Gemini to Q-Star - surveys 300+ papers and summarizes research developments to look at in the space of Generative AI; it covers computational challenges, scalability, real-world implications, and the potential for Gen AI to drive progress in fields like healthcare, finance, and education.	Paper, Tweet	
3) PromptBench - a unified library that supports comprehensive evaluation and analysis of LLMs; it consists of functionalities for prompt construction, prompt engineering, dataset and model loading, adversarial prompt attack, dynamic evaluation protocols, and analysis tools.	Paper, Tweet	
4) Exploiting Novel GPT-4 APIs - performs red-teaming on three functionalities exposed in the GPT-4 APIs: fine-tuning, function calling, and knowledge retrieval; Main findings: 1) fine-tuning on as few as 15 harmful examples or 100 benign examples can remove core safeguards from GPT-4, 2) GPT-4 Assistants divulge the function call schema and can be made to execute arbitrary function calls, and 3) knowledge retrieval can be hijacked by injecting instructions into retrieval documents.	Paper, Tweet	
5) Fact Recalling in LLMs - investigates how MLP layers implement a lookup table for factual recall; scopes the study on how early MLPs in Pythia 2.8B look up which of 3 different sports various athletes play; suggests that early MLP layers act as a lookup table and recommends thinking about the recall of factual knowledge in the model as multi-token embeddings.	Paper, Tweet	
6) Generative AI for Math - presents a diverse and high-quality math-centric corpus comprising of ~9.5 billion tokens to train foundation models.	Paper, Tweet	
7) Pricipled Instructions Are All You Need - introduces 26 guiding principles designed to streamline the process of querying and prompting large language models; applies these principles to conduct extensive experiments on LLaMA-1/2 (7B, 13B and 70B), GPT-3.5/4 to verify their effectiveness on instructions and prompts design.	Paper, Tweet	
8) A Survey of Reasoning with Foundation Models - provides a comprehensive survey of seminal foundational models for reasoning, highlighting the latest advancements in various reasoning tasks, methods, benchmarks, and potential future directions; also discusses how other developments like multimodal learning, autonomous agents, and super alignment accelerate and extend reasoning research.	Paper, Tweet	
9) Making LLMs Better at Dense Retrieval - proposes LLaRA which adapts an LLM for dense retrieval; it consists of two pretext tasks: EBAE (Embedding-Based Auto-Encoding) and EBAR (Embedding-Based Auto-Regression), where the text embeddings from LLM are used to reconstruct the tokens for the input sentence and predict the tokens for the next sentence, respectively; a LLaMa-2-7B was improved on benchmarks like MSMARCO and BEIR.	Paper	
10) Gemini vs GPT-4V - provides a comprehensive preliminary comparison and combination of vision-language models like Gemini and GPT-4V through several qualitative cases; finds that GPT-4V is precise and succinct in responses, while Gemini excels in providing detailed, expansive answers accompanied by relevant imagery and links.	Paper, Tweet	


Top ML Papers of the Week (December 18 - December 24)


Paper	Links	
1) Gemini’s Language Abilities - provides an impartial and reproducible study comparing several popular models like Gemini, GPT, and Mixtral; Gemini Pro achieves comparable but slightly lower accuracy than the current version of GPT 3.5 Turbo; Gemini and GPT were better than Mixtral.	Paper, Tweet	
2) PowerInfer - a high-speed inference engine for deploying LLMs locally; exploits the high locality in LLM inference to design a GPU-CPU hybrid inference engine; hot-activated neurons are preloaded onto the GPU for fast access, while cold-activated neurons (the majority) are computed on the CPU; this approach significantly reduces GPU memory demands and CPU-GPU data transfer.	Paper, Tweet	
3) Discovery of a New Family of Antibiotics with Graph Deep Learning - discovered a new structural class of antibiotics with explainable graph algorithms; the approach enables explainable deep learning guided discovery of structural classes of antibiotics which helps to provide chemical substructures that underlie antibiotic activity.	Paper, Tweet	
4) VideoPoet - introduces a large language model for zero-shot video generation; it’s capable of a variety of video generation tasks such as image-to-video and video stylization; trains an autoregressive model to learn across video, image, audio, and text modalities by using multiple tokenizers; shows that language models can synthesize and edit video with some degree of temporal consistency.	Paper, Tweet_	
5) Multimodal Agents as Smartphone Users - introduces an LLM-based multimodal agent framework to operate smartphone applications; learns to navigate new apps through autonomous exploration or observing human demonstrations; shows proficiency in handling diverse tasks across different applications like email, social media, shopping, editing tools, and more.	Paper, Tweet_	
6) LLM in a Flash - proposes an approach that efficiently runs LLMs that exceed the available DRAM capacity by storing the model parameters on flash memory but bringing them on demand to DRAM; enables running models up to twice the size of the available DRAM, with a 4-5x and 20-25x increase in inference speed compared to naive loading approaches in CPU and GPU, respectively.	Paper, Tweet_	
7) ReST Meets ReAct - proposes a ReAct-style agent with self-critique for improving on the task of long-form question answering; it shows that the agent can be improved through ReST-style (reinforced self-training) iterative fine-tuning on its reasoning traces; specifically, it uses growing-batch RL with AI feedback for continuous self-improvement and self-distillation; like a few other recent papers, it focuses on minimizing human involvement (i.e., doesn't rely on human-labeled training data); it generates synthetic data with self-improvement from AI feedback which can then be used to distill the agent into smaller models (1/2 orders magnitude) with comparable performance as the pre-trained agent.	Paper, Tweet_	
8) Adversarial Attacks on GPT-4 - uses a simple random search algorithm to implement adversarial attacks on GPT-4; it achieves jailbreaking by appending an adversarial suffix to an original request, then iteratively making slight random changes to the suffix, and keeping changes if it increases the log probability of the token “Sure” at the first position of the response.	Paper, Tweet_	
9) RAG for LLMs - an overview of all the retrieval augmented generation (RAG) research that has been happening.	Paper, Tweet_	
10) Findings of the BabyLLM Challenge - presents results for a new challenge that involves sample-efficient pretraining on a developmentally plausible corpus; the winning submission, which uses flashy LTG BERT, beat Llama 2 70B on 3/4 evals; other approaches that saw good results included data preprocessing or training on shorter context.	Paper, Tweet_	


Top ML Papers of the Week (December 11 - December 17)


Paper	Links	
1) LLMs for Discoveries in Mathematical Sciences - uses LLMs to search for new solutions in mathematics & computer science; proposes FunSearch which combines a pre-trained LLM with a systematic evaluator and iterates over them to evolve low-scoring programs into high-scoring ones discovering new knowledge; one of the key findings in this work is that safeguarding against LLM hallucinations is important to produce mathematical discoveries and other real-world problems.	Paper, Tweet	
2) Weak-to-strong Generalization - studies whether weak model supervision can elicit the full capabilities of stronger models; finds that when naively fine-tuning strong pretrained models on weak model generated labels they can perform better than their weak supervisors; reports that finetuning GPT-4 with a GPT-2-level supervisor it’s possible to recover close to GPT-3.5-level performance on NLP tasks.	Paper, Tweet	
3) Audiobox - a unified model based on flow-matching capable of generating various audio modalities; designs description-based and example-based prompting to enhance controllability and unify speech and sound generation paradigms; adapts a self-supervised infilling objective to pre-train on large quantities of unlabeled audio; performs well on speech and sound generation and unlocks new methods for generating audio with novel vocal and acoustic styles.	Paper, Tweet	
4) Mathematical LLMs - a survey on the progress of LLMs on mathematical tasks; covers papers and resources on LLM research around prompting techniques and tasks such as math word problem-solving and theorem proving.	Paper, Tweet	
5) Towards Fully Transparent Open-Source LLMs - proposes LLM360 to support open and collaborative AI research by making the end-to-end LLM training process transparent and reproducible; releases 7B parameter LLMs pre-trained from scratch, AMBER and CRYSTALCODER, including their training code, data, intermediate checkpoints, and analyses.	Paper, Tweet	
6) LLMs in Medicine - a comprehensive survey (analyzing 300+ papers) on LLMs in medicine; includes an overview of the principles, applications, and challenges faced by LLMs in medicine.	Paper, Tweet	
7) Beyond Human Data for LLMs - proposes an approach for self-training with feedback that can substantially reduce dependence on human-generated data; the model-generated data combined with a reward function improves the performance of LLMs on problem-solving tasks.	Paper, Tweet	
8) Gaussian-SLAM - a neural RGBD SLAM method capable of photorealistically reconstructing real-world scenes without compromising speed and efficiency; extends classical 3D Gaussians for scene representation to overcome the limitations of the previous methods.	Paper, Tweet	
9) Pearl - introduces a new production-ready RL agent software package that enables researchers and practitioners to develop RL AI agents that adapt to environments with limited observability, sparse feedback, and high stochasticity.	Paper, Tweet	
10) Quip - compresses trained model weights into a lower precision format to reduce memory requirements; the approach combines lattice codebooks with incoherence processing to create 2 bit quantized models; significantly closes the gap between 2 bit quantized LLMs and unquantized 16 bit models.	Paper, Tweet	


Top ML Papers of the Week (December 4 - December 10)


Paper	Links	
1) Gemini - a series of multimodal models with multimodal reasoning capabilities across text, images, video, audio, and code; claims to outperform human experts on MMLU, a popular benchmark to test the knowledge and problem-solving abilities of AI models; capabilities reported include multimodality, multilinguality, factuality, summarization, math/science, long-context, reasoning, and more.	Paper, Tweet	
2) EfficientSAM - a lightweight Segment Anything Model (SAM) that exhibits decent performance with largely reduced complexity; leverages masked autoencoders with 20x fewer parameters and 20x faster runtime; EfficientSAM performs within 2 points (44.4 AP vs 46.5 AP) of the original SAM model.	Paper, Tweet	
3) Magicoder - a series of fully open-source LLMs for code that close the gap with top code models while having no more than 7B parameters; trained on 75K synthetic instruction data; uses open-source references for the production of more diverse, realistic, high-quality, and controllable data; outperforms state-of-the-art code models with similar or even larger sizes on several coding benchmarks, including Python text-to-code generation, multilingual coding, and data-science program completion; MagicoderS-CL-7B based on CodeLlama surpasses ChatGPT on HumanEval+ (66.5 vs. 65.9 in pass@1).	Paper, Tweet	
4) LLMs on Graphs - a comprehensive overview that summarizes different scenarios where LLMs are used on graphs such as pure graphs, text-rich graphs, and text-paired graphs	Paper, Tweet	
5) Llama Guard - an LLM-based safeguard model that involves a small (Llama2-7B) customizable instruction-tuned model that can classify safety risks in prompts and responses for conversational AI agent use cases; the model can be leveraged in a zero-shot or few-shot way if you need to adapt it to a different safety risk taxonomy that meets the requirements for a target use case; it can also be fine-tune on a specific dataset to adapt to a new taxonomy.	Paper, Tweet	
6) Human-Centered Loss Functions - proposes an approach called Kahneman-Tversky Optimization (KTO) that matches or exceeds DPO performance methods at scales from 1B to 30B; KTO maximizes the utility of LLM generations instead of maximizing the log-likelihood of preferences as most current methods do.	Paper, Tweet	
7) Chain of Code - a simple extension of the chain-of-thought approach that improves LM code-driven reasoning; it encourages LMs to format semantic sub-tasks in a program as pseudocode that the interpreter can explicitly catch undefined behavior and hand off to simulate with an LLM; on BIG-Bench Hard, Chain of Code achieves 84%, a gain of 12% over Chain of Thought.	Paper, Tweet	
8) Data Management For LLMs - an overview of current research in data management within both the pretraining and supervised fine-tuning stages of LLMs; it covers different aspects of data management strategy design: data quantity, data quality, domain/task composition, and more.	Paper, Tweet	
9) 8RankZephyr* - an open-source LLM for listwise zero-shot reranking that bridges the effectiveness gap with GPT-4 and in some cases surpasses the proprietary model; it outperforms GPT-4 on the NovelEval test set, comprising queries and passages past its training period, which addresses concerns about data contamination.	Paper, Tweet	
10) The Efficiency Spectrum of LLMs - a comprehensive review of algorithmic advancements aimed at improving LLM efficiency; covers various topics related to efficiency, including scaling laws, data utilization, architectural innovations, training and tuning strategies, and inference techniques.	Paper, Tweet	


Top ML Papers of the Week (November 27 - December 3)


Paper	Links	
1) GNoME - a new AI system for material design that finds 2.2 million new crystals, including 380,000 stable materials; presents a new deep learning tool that increases the speed and efficiency of discovery by predicting the stability of new materials.	Paper, Tweet	
2) Open-Source LLMs vs. ChatGPT - provides an exhaustive overview of tasks where open-source LLMs claim to be on par or better than ChatGPT.	Paper, Tweet	
3) Adversarial Diffusion Distillation - a novel training approach that efficiently samples large-scale foundation image diffusion models in just 1-4 steps while maintaining high image quality; combines score distillation and an adversarial loss to ensure high image fidelity even in the low-step regime of one or two sampling steps; reaches performance of state-of-the-art diffusion models in only four steps.	Paper, Tweet	
4) Seamless - a family of research models that enable end-to-end expressive cross-lingual communication in a streaming fashion; introduces an improved SeamlssM4T model trained on more low-resource language data; also applies red-teaming effort for safer multimodal machine translation.	Paper, Tweet	
5) MEDITRON-70B - a suite of open-source LLMs with 7B and 70B parameters adapted to the medical domain; builds on Llama-2 and extends pretraining on a curated medical corpus; MEDITRON-70B outperforms GPT-3.5 and Med-PaLM and is within 5% of GPT-4 and 10% of Med-PaLM-2.	Paper, Tweet	
6) Foundation Models Outcompeting Special-Purpose Tuning - performs a systematic exploration of prompt engineering to boost the performance of LLMs on medical question answering; uses prompt engineering methods that are general purpose and make no use of domain expertise; prompt engineering led to enhancing GPT-4’s performance and achieves state-of-the-art results on nine benchmark datasets in the MultiMedQA suite.	Paper, Tweet	
7) UniIR - a unified instruction-guided multimodal retriever that handles eight retrieval tasks across modalities; can generalize to unseen retrieval tasks and achieves robust performance across existing datasets and zero-shot generalization to new tasks; presents a multimodal retrieval benchmark to help standardize the evaluation of multimodal information retrieval.	Paper, Tweet	
8) Safe Deployment of Generative AI - argues that to protect people’s privacy, medical professionals, not commercial interests, must drive the development and deployment of such models.	Paper, Tweet	
9) On Bringing Robots Home - introduces Dobb-E, an affordable and versatile general-purpose system for learning robotic manipulation within household settings; Dobbe-E can learn new tasks with only 5 minutes of user demonstrations; experiments reveal unique challenges absent or ignored in lab robotics, including effects of strong shadows, variable demonstration quality by non-expert users, among others.	Paper, Tweet	
10) Translatotron 3 - proposes an unsupervised approach to speech-to-speech translation that can learn from monolingual data alone; combines masked autoencoder, unsupervised embedding mapping, and back-translation; results show that the model outperforms a baseline cascade system and showcases its capability to retain para-/non-linguistic such as pauses, speaking rates, and speaker identity.	Paper, Tweet	


Top ML Papers of the Week (November 20 - November 26)


Paper	Links	
1) System 2 Attention - leverages the reasoning and instruction following capabilities of LLMs to decide what to attend to; it regenerates input context to only include relevant portions before attending to the regenerated context to elicit the final response from the model; increases factuality and outperforms standard attention-based LLMs on tasks such as QA and math world problems.	Paper, Tweet	
2) Advancing Long-Context LLMs - an overview of the methodologies for enhancing Transformer architecture modules that optimize long-context capabilities across all stages from pre-training to inference.	Paper, Tweet	
3) Parallel Speculative Sampling - approach to reduce inference time of LLMs based on a variant of speculative sampling and parallel decoding; achieves significant speed-ups (up to 30%) by only learning as little as O(d_emb) additional parameters.	Paper, Tweet	
4) Mirasol3B - a multimodal model for learning across audio, video, and text which decouples the multimodal modeling into separate, focused autoregressive models; the inputs are processed according to the modalities; this approach can handle longer videos compared to other models and it outperforms state-of-the-art approach on video QA, long video QA, and audio-video-text benchmark.	Paper, Tweet	
5) Teaching Small LMs To Reason - proposes an approach to teach smaller language models to reason; specifically, the LM is thought to use reasoning techniques, such as step-by-step processing, recall-then-generate, recall-reason-generate, extract-generate, and direct-answer methods; outperforms models of similar size and attains performance levels similar or better to those of models 5-10x larger, as assessed on complex tasks that test advanced reasoning abilities in zero-shot settings.	Paper, Tweet	
6) GPQA - proposes a graduate-level Google-proof QA benchmark consisting of 448 multiple-choice questions written by domain experts in biology, physics, and chemistry; the strongest GPT-4 based baseline achieves 39% accuracy; this benchmark offers scalable oversight experiments that can help obtain reliable and truthful information from modern AI systems that surpass human capabilities.	Paper, Tweet	
7) The Hitchhiker’s Guide From Chain-of-Thought Reasoning to Language Agents - summary of CoT reasoning, foundational mechanics underpinning CoT techniques, and their application to language agent frameworks.	Paper, Tweet	
8) GAIA - a benchmark for general AI assistants consisting of real-world questions that require a set of fundamental abilities such as reasoning, multimodal handling, web browsing, and generally tool-use proficiency; shows that human respondents obtain 92% vs. 15% for GPT-4 equipped with plugins.	Paper, Tweet	
9) LLMs as Collaborators for Medical Reasoning - proposes a collaborative multi-round framework for the medical domain that leverages role-playing LLM-based agents to enhance LLM proficiency and reasoning capabilities.	Paper, Tweet	
10) TÜLU 2 - presents a suite of improved TÜLU models for advancing the understanding and best practices of adapting pretrained language models to downstream tasks and user preferences; TÜLU 2 suite achieves state-of-the-art performance among open models and matches or exceeds the performance of GPT-3.5-turbo-0301 on several benchmarks.	Paper, Tweet	


Top ML Papers of the Week (November 13 - November 19)


Paper	Links	
1) Emu Video and Emu Edit - present new models for controlled image editing and text-to-video generation based on diffusion models; Emu Video can generate high-quality video by using text-only, image-only, or combined text and image inputs; Emu Edit enables free-form editing through text instructions.	Paper, Tweet	
2) Chain-of-Note - an approach to improve the robustness and reliability of retrieval-augmented language models in facing noisy, irrelevant documents and in handling unknown scenarios; CoN generates sequential reading notes for the retrieved documents, enabling an evaluation of their relevance to the given question and integrating this information to formulate the final answer; CoN significantly outperforms standard retrieval-augmented language models and achieves an average improvement of +7.9 in EM score given entirely noisy retrieved documents and +10.5 in rejection rates for real-time questions that fall outside the pre-training knowledge scope.	Paper, Tweet	
3) LLMs for Scientific Discovery - explores the impact of large language models, particularly GPT-4, across various scientific fields including drug discovery, biology, and computational chemistry; assesses GPT-4's understanding of complex scientific concepts, its problem-solving capabilities, and its potential to advance scientific research through expert-driven case assessments and benchmark testing.	Paper, Tweet	
4) Fine-Tuning LLMs for Factuality - fine-tunes language model for factuality without requiring human labeling; it learns from automatically generated factuality preference rankings and targets open-ended generation settings; it significantly improves the factuality of Llama-2 on held-out topics compared with RLHF or decoding strategies targeted at factuality.	Paper, Tweet	
5) Contrastive CoT Prompting - proposes a contrastive chain of thought method to enhance language model reasoning; the approach provides both valid and invalid reasoning demonstrations, to guide the model to reason step-by-step while reducing reasoning mistakes; also proposes an automatic method to construct contrastive demonstrations and demonstrates improvements over CoT prompting.	Paper, Tweet	
6) A Survey on Language Models for Code - provides an overview of LLMs for code, including a review of 50+ models, 30+ evaluation tasks, and 500 related works.	Paper, Tweet	
7) JARVIS-1 - an open-world agent that can perceive multimodal input	Paper, Tweet	
8) Learning to Filter Context for RAG - proposes a method that improves the quality of the context provided to the generator via two steps: 1) identifying useful context based on lexical and information-theoretic approaches, and 2) training context filtering models that can filter retrieved contexts at inference; outperforms existing approaches on extractive question answering	Paper, Tweet	
9) MART - proposes an approach for improving LLM safety with multi-round automatic red-teaming; incorporates automatic adversarial prompt writing and safe response generation, which increases red-teaming scalability and the safety of LLMs; violation rate of an LLM with limited safety alignment reduces up to 84.7% after 4 rounds of MART, achieving comparable performance to LLMs with extensive adversarial prompt writing.	Paper, Tweet	
10) LLMs can Deceive Users - explores the use of an autonomous stock trading agent powered by LLMs; finds that the agent acts upon insider tips and hides the reason behind the trading decision; shows that helpful and safe LLMs can strategically deceive users in a realistic situation without direction instructions or training for deception.	Paper, Tweet	


Top ML Papers of the Week (November 6 - November 12)


Paper	Links	
1) Hallucination in LLMs - a comprehensive survey	Paper, Tweet	
2) Simplifying Transformer Blocks - explores simplifying the transformer block and finds that many block components can be removed with no loss of training speed; using different architectures like autoregressive decoder-only and BERT encoder-only models, the simplified blocks emulate per-update training speed and performance of standard transformers, and even achieve 15% faster training throughput with fewer parameters	Paper, Tweet	
3) Understanding In-Context Learning Abilities in Transformers - investigates how effectively transformers can bridge between pretraining data mixture to identify and learn new tasks in-context which are both inside and outside the pretraining distribution; in the regimes studied, there is limited evidence that the models’ in-context learning behavior is capable of generalizing beyond their pretraining data.	Paper, Tweet	
4) MusicGen - a single-stage transformer-based LLM that operates over several streams of compressed discrete music representation; it can generate high-quality samples	Paper, Tweet	
5) AltUp - a method that makes it possible to take advantage of increasing scale and capacity in Transformer models without increasing the computational cost; achieved by working on a subblock of the widened representation at each layer and using a predict-and-correct mechanism to update the inactivated blocks; it widens the learn representation while only incurring a negligible increase in latency.	Paper, Tweet	
6) Rephrase and Respond - an effective prompting method that uses LLMs to rephrase and expand questions posed by humans to improve overall performance; it can improve the performance of different models across a wide range of tasks; the approach can be combined with chain-of-thought to improve performance further.	Paper, Tweet	
7) On the Road with GPT-4V(ision) - provides an exhaustive evaluation of the latest state-of-the-art visual language model, GPT-4V(vision), and its application in autonomous driving; the model demonstrates superior performance in scene understanding and causal reasoning compared to existing autonomous systems.	Paper, Tweet	
8) GPT4All - outlines technical details of the GPT4All model family along with the open-source repository that aims to democratize access to LLMs.	Paper, Tweet	
9) S-LoRA - an approach that enables the scalable serving of many LoRA adapters; it stores all adapters in main memory and fetches adapters of currently running queries to the GPU memory; employs novel tensor parallelism strategy and highly optimized custom CUDA kernels for heterogenous batching of LoRA computation; improves throughput by 4x, when compared to other solutions, and increases the number of served adapters by several orders of magnitude.	Paper, Tweet	
10) FreshLLMs - proposes a dynamic QA benchmark	Paper, Tweet	


Top ML Papers of the Week (October 30 - November 5)


Paper	Links	
1) MetNet-3 - a state-of-the-art neural weather model that extends both the lead time range and the variables that an observation-based model can predict well; learns from both dense and sparse data sensors and makes predictions up to 24 hours ahead for precipitation, wind, temperature, and dew point.	Paper, Tweet	
2) Evaluating LLMs - a comprehensive survey	Paper, Tweet	
3) Battle of the Backbones - a large benchmarking framework for a diverse suite of computer vision tasks; find that while vision transformers	Paper, Tweet	
4) LLMs for Chip Design - proposes using LLMs for industrial chip design by leveraging domain adaptation techniques; evaluates different applications for chip design such as assistant chatbot, electronic design automation, and bug summarization; domain adaptation significantly improves performance over general-purpose models on a variety of design tasks; using a domain-adapted LLM for RAG further improves answer quality.	Paper, Tweet	
5) Efficient Context Window Extension of LLMs - proposes a compute-efficient method for efficiently extending the context window of LLMs beyond what it was pretrained on; extrapolates beyond the limited context of a fine-tuning dataset and models have been reproduced up to 128K context length.	Paper, Tweet	
6) Open DAC 2023 - introduces a dataset consisting of more than 38M density functional theory	Paper, Tweet	
7) Symmetry in Machine Learning - presents a unified and methodological framework to enforce, discover, and promote symmetry in machine learning; also discusses how these ideas can be applied to ML models such as multilayer perceptions and basis function regression.	Paper, Tweet	
8) Next Generation AlphaFold - reports progress on a new iteration of AlphaFold that greatly expands its range of applicability; shows capabilities of joint structure prediction of complexes including proteins, nucleic acids, small molecules, ions, and modified residue; demonstrates greater accuracy on protein-nucleic acid interactions than specialists predictors.	Paper, Tweet	
9) Enhancing LLMs by Emotion Stimuli - explores the ability of LLMs to understand emotional stimuli; conducts automatic experiments on 45 tasks using various LLMs, including Flan-T5-Large, Vicuna, Llama 2, BLOOM, ChatGPT, and GPT-4; the tasks span deterministic and generative applications that represent comprehensive evaluation scenarios; experimental results show that LLMs have a grasp of emotional intelligence.	Paper, Tweet	
10) FP8-LM - finds that when training FP8 LLMs most variables, such as gradients and optimizer states, in LLM training, can employ low-precision data formats without compromising model accuracy and requiring no changes to hyper-parameter.	Paper, Tweet	


Top ML Papers of the Week (October 23 - October 29)


Paper	Links	
1) Zephyr LLM - a 7B parameter model with competitive performance to ChatGPT on AlpacaEval; applies distilled supervised fine-tuning to improve task accuracy and distilled direct performance optimization on AI feedback data to better align the model; shows performance comparable to 70B-parameter chat models aligned with human feedback.	Paper, Tweet	
2) Fact-checking with LLMs - investigates the fact-checking capabilities of LLMs like GPT-4; results show the enhanced prowess of LLMs when equipped with contextual information; GPT4 outperforms GPT-3, but accuracy varies based on query language and claim veracity; while LLMs show promise in fact-checking, they demonstrate inconsistent accuracy.	Paper, Tweet	
3) Matryoshka Diffusion Models - introduces an end-to-end framework for high-resolution image and video synthesis; involves a diffusion process that denoises inputs at multiple resolutions jointly and uses a NestedUNet architecture; enables a progressive training schedule from lower to higher resolutions leading to improvements in optimization for high-resolution generation.	Paper, Tweet	
4) Spectron - a new approach for spoken language modeling trained end-to-end to directly process spectrograms; it can be fine-tuned to generate high-quality accurate spoken language; the method surpasses existing spoken language models in speaker preservation and semantic coherence.	Paper, Tweet	
5) LLMs Meet New Knowledge - presents a benchmark to assess LLMs' abilities in knowledge understanding, differentiation, and association; benchmark results show	Paper, Tweet	
6) Detecting Pretraining Data from LLMs - explores the problem of pretraining data detection which aims to determine if a black box model was trained on a given text; proposes a detection method named Min-K% Prob as an effective tool for benchmark example contamination detection, privacy auditing of machine unlearning, and copyrighted text detection in LM’s pertaining data.	Paper, Tweet	
7) ConvNets Match Vision Transformers - evaluates a performant ConvNet architecture pretrained on JFT-4B at scale; observes a log-log scaling law between the held out loss and compute budget; after fine-tuning on ImageNet, NFNets match the reported performance of Vision Transformers with comparable compute budgets.	Paper, Tweet	
8) CommonCanvas - a dataset of Creative-Commons-licensed	Paper, Tweet	
9) Managing AI Risks - a short paper outlining risks from upcoming and advanced AI systems, including an examination of social harms, malicious uses, and other potential societal issues emerging from the rapid adoption of autonomous AI systems.	Paper, Tweet	
10) Branch-Solve-Merge Reasoning in LLMs - an LLM program that consists of branch, solve, and merge modules parameterized with specific prompts to the base LLM; this enables an LLM to plan a decomposition of task into multiple parallel sub-tasks, independently solve them, and fuse solutions to the sub-tasks; improves evaluation correctness and consistency for multiple LLMs.	Paper, Tweet	


Top ML Papers of the Week (October 16 - October 22)


Paper	Links	
1) Llemma - an LLM for mathematics which is based on continued pretraining from Code Llama on the Proof-Pile-2 dataset; the dataset involves scientific paper, web data containing mathematics, and mathematical code; Llemma outperforms open base models and the unreleased Minerva on the MATH benchmark; the model is released, including dataset and code to replicate experiments.	Paper, Tweet	
2) LLMs for Software Engineering - a comprehensive survey of LLMs for software engineering, including open research and technical challenges.	Paper, Tweet	
3) Self-RAG - presents a new retrieval-augmented framework that enhances an LM’s quality and factuality through retrieval and self-reflection; trains an LM that adaptively retrieves passages on demand, and generates and reflects on the passages and its own generations using special reflection tokens; it significantly outperforms SoTA LLMs	Paper, Tweet	
4) Retrieval-Augmentation for Long-form Question Answering - explores retrieval-augmented language models on long-form question answering; finds that retrieval is an important component but evidence documents should be carefully added to the LLM; finds that attribution error happens more frequently when retrieved documents lack sufficient information/evidence for answering the question.	Paper, Tweet	
5) GenBench - presents a framework for characterizing and understanding generalization research in NLP; involves a meta-analysis of 543 papers and a set of tools to explore and better understand generalization studies.	Paper, Tweet	
6) A Study of LLM-Generated Self-Explanations - assesses an LLM's capability to self-generate feature attribution explanations; self-explanation is useful to improve performance and truthfulness in LLMs; this capability can be used together with chain-of-thought prompting.	Paper, Tweet	
7) OpenAgents - an open platform for using and hosting language agents in the wild; includes three agents, including a Data Agent for data analysis, a Plugins Agent with 200+ daily API tools, and a Web Agent for autonomous web browsing.	Paper, Tweet	
8) Eliciting Human Preferences with LLMs - uses language models to guide the task specification process and a learning framework to help models elicit and infer intended behavior through free-form, language-based interaction with users; shows that by generating open-ended questions, the system generates responses that are more informative than user-written prompts.	Paper, Tweet	
9) AutoMix - an approach to route queries to LLMs based on the correctness of smaller language models	Paper, Tweet	
10) Video Language Planning - enables synthesizing complex long-horizon video plans across robotics domains; the proposed algorithm involves a tree search procedure that trains vision-language models to serve as policies and value functions, and text-to-video models as dynamic models.	Paper, Tweet	


Top ML Papers of the Week (October 9 - October 15)


Paper	Links	
1) Ring Attention - a memory-efficient approach that leverages blockwise computation of self-attention to distribute long sequences across multiple devices to overcome the memory limitations inherent in Transformer architectures, enabling handling of longer sequences during training and inference; enables scaling the context length with the number of devices while maintaining performance, exceeding context length of 100 million without attention approximations.	Paper, Tweet	
2) Universal Simulator - applies generative modeling to learn a universal simulator of real-world interactions; can emulate how humans and agents interact with the world by simulating the visual outcome of high instruction and low-level controls; the system can be used to train vision-language planners, low-level reinforcement learning policies, and even for systems that perform video captioning.	Paper, Tweet	
3) Overview of Factuality in LLMs - a survey of factuality in LLMs providing insights into how to evaluate factuality in LLMs and how to enhance it.	Paper, Tweet	
4) LLMs can Learn Rules - presents a two-stage framework that learns a rule library for reasoning with LLMs; in the first stage	Paper, Tweet	
5) Meta Chain-of-Thought Prompting - a generalizable chain-of-thought	Paper, Tweet	
6) A Survey of LLMs for Healthcare - a comprehensive overview of LLMs applied to the healthcare domain.	Paper, Tweet	
7) Improving Retrieval-Augmented LMs with Compressors - presents two approaches to compress retrieved documents into text summaries before pre-pending them in-context: 1) extractive compressor - selects useful sentences from retrieved documents 2) abstractive compressor - generates summaries by synthesizing information from multiple documents; achieves a compression rate of as low as 6% with minimal loss in performance on language modeling tasks and open domain question answering tasks; the proposed training scheme performs selective augmentation which helps to generate empty summaries when retrieved docs are irrelevant or unhelpful for a task.	Paper, Tweet	
8) Instruct-Retro - introduces Retro 48B, the largest LLM pretrained with retrieval; continues pretraining a 43B parameter GPT model on an additional 100B tokens by retrieving from 1.2T tokens	Paper, Tweet	
9) MemWalker - a method to enhance long-text understanding by treating the LLM as an interactive agent that can decide how to read the text via iterative prompting; it first processes long context into a tree of summer nodes and reads in a query to traverse the tree, seeking relevant information and crafting a suitable response; this process is achieved through reasoning and enables effective reading and enhances explainability through reasoning steps.	Paper, Tweet	
10) Toward Language Agent Fine-tuning - explores the direction of fine-tuning LLMs to obtain language agents; finds that language agents consistently improved after fine-tuning their backbone language model; claims that fine-tuning a Llama2-7B with 500 agent trajectories	Paper, Tweet	


Top ML Papers of the Week (October 2 - October 8)


Paper	Links	
1) LLMs Represent Space and Time - discovers that LLMs learn linear representations of space and time across multiple scales; the representations are robust to prompt variations and unified across different entity types; demonstrate that LLMs acquire fundamental structured knowledge such as space and time, claiming that language models learn beyond superficial statistics, but literal world models.	Paper, Tweet	
2) Retrieval meets Long Context LLMs - compares retrieval augmentation and long-context windows for downstream tasks to investigate if the methods can be combined to get the best of both worlds; an LLM with a 4K context window using simple RAG can achieve comparable performance to a fine-tuned LLM with 16K context; retrieval can significantly improve the performance of LLMs regardless of their extended context window sizes; a retrieval-augmented LLaMA2-70B with a 32K context window outperforms GPT-3.5-turbo-16k on seven long context tasks including question answering and query-based summarization.	Paper, Tweet	
3) StreamingLLM - a framework that enables efficient streaming LLMs with attention sinks, a phenomenon where the KV states of initial tokens will largely recover the performance of window attention; the emergence of the attention sink is due to strong attention scores towards the initial tokens; this approach enables LLMs trained with finite length attention windows to generalize to infinite sequence length without any additional fine-tuning.	Paper, Tweet	
4) Neural Developmental Programs - proposes to use neural networks that self-assemble through a developmental process that mirrors properties of embryonic development in biological organisms	Paper, Tweet	
5) The Dawn of LMMs - a comprehensive analysis of GPT-4V to deepen the understanding of large multimodal models	Paper, Tweet	
6) Training LLMs with Pause Tokens - performs training and inference on LLMs with a learnable token which helps to delay the model's answer generation and attain performance gains on general understanding tasks of Commonsense QA and math word problem-solving; experiments show that this is only beneficial provided that the delay is introduced in both pertaining and downstream fine-tuning.	Paper, Tweet	
7) Recursively Self-Improving Code Generation - proposes the use of a language model-infused scaffolding program to recursively improve itself; a seed improver first improves an input program that returns the best solution which is then further tasked to improve itself; shows that the GPT-4 models can write code that can call itself to improve itself.	Paper, Tweet	
8) Retrieval-Augmented Dual Instruction Tuning - proposes a lightweight fine-tuning method to retrofit LLMs with retrieval capabilities; it involves a 2-step approach: 1) updates a pretrained LM to better use the retrieved information 2) updates the retriever to return more relevant results, as preferred by the LM Results show that fine-tuning over tasks that require both knowledge utilization and contextual awareness, each stage leads to additional gains; a 65B model achieves state-of-the-art results on a range of knowledge-intensive zero- and few-shot learning benchmarks; it outperforms existing retrieval-augmented language approaches by up to +8.9% in zero-shot and +1.4% in 5-shot.	Paper, Tweet	
9) KOSMOG-G - a model that performs high-fidelity zero-shot image generation from generalized vision-language input that spans multiple images; extends zero-shot subject-driven image generation to multi-entity scenarios; allows the replacement of CLIP, unlocking new applications with other U-Net techniques such as ControlNet and LoRA.	Paper, Tweet	
10) Analogical Prompting - a new prompting approach to automatically guide the reasoning process of LLMs; the approach is different from chain-of-thought in that it doesn’t require labeled exemplars of the reasoning process; the approach is inspired by analogical reasoning and prompts LMs to self-generate relevant exemplars or knowledge in the context.	Paper, Tweet	


Top ML Papers of the Week (September 25 - October 1)


Paper	Links	
1) The Reversal Curse - finds that LLMs trained on sentences of the form “A is B” will not automatically generalize to the reverse direction “B is A”, i.e., the Reversal Curse; shows the effect through finetuning LLMs on fictitious statements and demonstrating its robustness across model sizes and model families.	Paper, Tweet	
2) Effective Long-Context Scaling with LLMs - propose a 70B variant that can already surpass gpt-3.5-turbo-16k’s overall performance on a suite of long-context tasks. This involves a cost-effective instruction tuning procedure that does not require human-annotated long instruction data.	Paper, Tweet	
3) Graph Neural Prompting with LLMs - proposes a plug-and-play method to assist pre-trained LLMs in learning beneficial knowledge from knowledge graphs	Paper, Tweet	
4) Vision Transformers Need Registers - identifies artifacts in feature maps of vision transformer networks that are repurposed for internal computations; this work proposes a solution to provide additional tokens to the input sequence to fill that role; the solution fixes the problem, leads to smoother feature and attention maps, and sets new state-of-the-art results on dense visual prediction tasks.	Paper, Tweet	
5) Boolformer - presents the first Transformer architecture trained to perform end-to-end symbolic regression of Boolean functions; it can predict compact formulas for complex functions and be applied to modeling the dynamics of gene regulatory networks.	Paper, Tweet	
6) LlaVA-RLHF - adapts factually augmented RLHF to aligning large multimodal models; this approach alleviates the reward hacking in RLHF and improves performance on the LlaVA-Bench dataset with the 94% performance level of the text-only GPT-4.	Paper, Tweet	
7) LLM Alignment Survey - a comprehensive survey paper on LLM alignment; topics include Outer Alignment, Inner Alignment, Mechanistic Interpretability, Attacks on Aligned LLMs, Alignment Evaluation, Future Directions, and Discussions.	Paper, Tweet	
8) Qwen LLM - proposes a series of LLMs demonstrating the strength of RLHF on tasks involving tool use and planning capabilities for creating language agents.	Paper, Tweet	
9) MentalLlaMa - an open-source LLM series for interpretable mental health analysis with instruction-following capability; it also proposes a multi-task and multi-source interpretable mental health instruction dataset on social media with 105K data samples.	Paper, Tweet	
10) Logical Chain-of-Thought in LLMs - a new neurosymbolic framework to improve zero-shot chain-of-thought reasoning in LLMs; leverages principles from symbolic logic to verify and revise reasoning processes to improve the reasoning capabilities of LLMs.	Paper, Tweet	


Top ML Papers of the Week (September 18 - September 24)


Paper	Links	
1) AlphaMissense - an AI model classifying missense variants to help pinpoint the cause of diseases; the model is used to develop a catalogue of genetic mutations; it can categorize 89% of all 71 million possible missense variants as either likely pathogenic or likely benign.	Paper, Tweet	
2) Chain-of-Verification reduces Hallucination in LLMs - develops a method to enable LLMs to "deliberate" on responses to correct mistakes; include the following steps: 1) draft initial response, 2) plan verification questions to fact-check the draft, 3) answer questions independently to avoid bias from other responses, and 4) generate a final verified response.	Paper, Tweet	
3) Contrastive Decoding Improves Reasoning in Large Language Models - shows that contrastive decoding leads Llama-65B to outperform Llama 2 and other models on commonsense reasoning and reasoning benchmarks.	Paper, Tweet	
4) LongLoRA - an efficient fine-tuning approach to significantly extend the context windows of pre-trained LLMs; implements shift short attention, a substitute that approximates the standard self-attention pattern during training; it has less GPU memory cost and training time compared to full fine-tuning while not compromising accuracy.	Paper, Tweet	
5) LLMs for Generating Structured Data - studies the use of LLMs for generating complex structured data; proposes a structure-aware fine-tuning method, applied to Llama-7B, which significantly outperform other model like GPT-3.5/4 and Vicuna-13B.	Paper, Tweet	
6) LMSYS-Chat-1M - a large-scale dataset containing 1 million real-world conversations with 25 state-of-the-art LLM; it is collected from 210K unique IP addresses on the Vincuna demo and Chatbot Arena website.	Paper, Tweet	
7) Language Modeling is Compression - evaluates the compression capabilities of LLMs; it investigates how and why compression and prediction are equivalent; shows that LLMs are powerful general-purpose compressors due to their in-context learning abilities; finds that Chinchilla 70B compresses ImageNet patches to 43.4% and LibriSpeech samples to 16.4% of their raw size, beating domain-specific compressors like PNG	Paper, Tweet	
8) Compositional Foundation Models - proposes foundation models that leverage multiple expert foundation models trained on language, vision, and action data to solve long-horizon goals.	Paper, Tweet	
9) LLMs for IT Operations - proposes OWL, an LLM for IT operations tuned using a self-instruct strategy based on IT-related tasks; it discusses how to collect a quality instruction dataset and how to put together a benchmark.	Paper, Tweet	
10) KOSMOS-2.5 - a multimodal model for machine reading of text-intensive images, capable of document-level text generation and image-to-markdown text generation.	Paper, Tweet	


Top ML Papers of the Week (September 11 - September 17)


Paper	Links	
1) Textbooks Are All You Need II - a new 1.3 billion parameter model trained on 30 billion tokens; the dataset consists of "textbook-quality" synthetically generated data; phi-1.5 competes or outperforms other larger models on reasoning tasks suggesting that data quality plays a more important role than previously thought.	Paper, Tweet	
2) The Rise and Potential of LLM Based Agents - a comprehensive overview of LLM based agents; covers from how to construct these agents to how to harness them for good.	Paper, Tweet	
3) EvoDiff - combines evolutionary-scale data with diffusion models for controllable protein generation in sequence space; it can generate proteins inaccessible to structure-based models.	Paper, Tweet	
4) LLMs Can Align Themselves without Finetuning? - discovers that by integrating self-evaluation and rewind mechanisms, unaligned LLMs can directly produce responses consistent with human preferences via self-boosting.	Paper, Tweet	
5) Robot Parkour Learning - presents a system for learning end-to-end vision-based parkour policy which is transferred to a quadrupedal robot using its ecocentric depth camera; shows that low-cost robots can automatically select and execute parkour skills in a real-world environment.	Paper, Tweet	
6) A Survey of Hallucination in LLMs - classifies different types of hallucination phenomena and provides evaluation criteria for assessing hallucination along with mitigation strategies.	Paper, Tweet	
7) Agents - an open-source library for building autonomous language agents including support for features like planning, memory, tool usage, multi-agent communication, and more.	Paper, Tweet	
8) Radiology-Llama2: Best-in-Class LLM for Radiology - presents an LLM based on Llama 2 tailored for radiology; it's tuned on a large dataset of radiology reports to generate coherent and clinically useful impressions from radiology findings.	Paper, Tweet	
9) Communicative Agents for Software Development - presents ChatDev, a virtual chat-powered software development company mirroring the waterfall model; shows the efficacy of the agent in software generation, even completing the entire software development process in less than seven minutes for less than one dollar.	Paper, Tweet	
10) MAmmoTH - a series of open-source LLMs tailored for general math problem-solving; the models are trained on a curated instruction tuning dataset and outperform existing open-source models on several mathematical reasoning datasets.	Paper, Tweet	


Top ML Papers of the Week (September 4 - September 10)


Paper	Links	
1) Transformers as SVMs - finds that the optimization geometry of self-attention in Transformers exhibits a connection to hard-margin SVM problems; also finds that gradient descent applied without early-stopping leads to implicit regularization and convergence of self-attention; this work has the potential to deepen the understanding of language models.	Paper	
2) Scaling RLHF with AI Feedback - tests whether RLAIF is a suitable alternative to RLHF by comparing the efficacy of human vs. AI feedback; uses different techniques to generate AI labels and conduct scaling studies to report optimal settings for generating aligned preferences; the main finding is that on the task of summarization, human evaluators prefer generations from both RLAIF and RLHF over a baseline SFT model in ∼70% of cases.	Paper, Tweet	
3) GPT Solves Math Problems Without a Calculator - shows that with sufficient training data, a 2B language model can perform multi-digit arithmetic operations with 100% accuracy and without data leakage; it’s also competitive with GPT-4 on 5K samples Chinese math problem test set when fine-tuned from GLM-10B on a dataset containing additional multi-step arithmetic operations and detailed math problems.	Paper, Tweet	
4) LLMs as Optimizers - an approach where the optimization problem is described in natural language; an LLM is then instructed to iteratively generate new solutions based on the defined problem and previously found solutions; at each optimization step, the goal is to generate new prompts that increase test accuracy based on the trajectory of previously generated prompts; the optimized prompts outperform human-designed prompts on GSM8K and Big-Bench Hard, sometimes by over 50%	Paper, Tweet	
5) Multi-modality Instruction Tuning - presents ImageBind-LLM, a multimodality instruction tuning method of LLMs via ImageBind; this model can respond to instructions of diverse modalities such as audio, 3D point clouds, and video, including high language generation quality; this is achieved by aligning ImageBind’s visual encoder with an LLM via learnable bind network.	Paper, Tweet	
6) Explaining Grokking - aims to explain grokking behavior in neural networks; specifically, it predicts and shows two novel behaviors: the first is ungrokking where a model goes from perfect generalization to memorization when trained further on a smaller dataset than the critical threshold; the second is semi-grokking where a network demonstrates grokking-like transition when training a randomly initialized network on the critical dataset size.	Paper, Tweet	
7) Overview of AI Deception - provides a survey of empirical examples of AI deception.	Paper, Tweet	
8) FLM-101B - a new open LLM called FLM-101B with 101B parameters and 0.31TB tokens which can be trained on a $100K budget; the authors analyze different growth strategies, growing the number of parameters from smaller sizes to large ones. They ultimately employ an aggressive strategy that reduces costs by >50%. In other words, three models are trained sequentially with each model inheriting knowledge from its smaller predecessor	Paper, Tweet	
9) Cognitive Architecture for Language Agents - proposes a systematic framework for understanding and building fully-fledged language agents drawing parallels from production systems and cognitive architectures; it systematizes diverse methods for LLM-based reasoning, grounding, learning, and decision making as instantiations of language agents in the framework.	Paper, Tweet	
10) Q-Transformer - a scalable RL method for training multi-task policies from large offline datasets leveraging human demonstrations and autonomously collected data; shows good performance on a large diverse real-world robotic manipulation task suite.	Paper, Tweet	


Top ML Papers of the Week (August 28 - September 3)


Paper	Links	
1) Large Language and Speech Model - proposes a large language and speech model trained with cross-modal conversational abilities that supports speech-and-language instruction enabling more natural interactions with AI systems.	Paper, Tweet	
2) SAM-Med2D - applies segment anything models	Paper, Tweet	
3) Vector Search with OpenAI Embeddings - suggests that “from a cost–benefit analysis, there does not appear to be a compelling reason to introduce a dedicated vector store into a modern “AI stack” for search since such applications have already received substantial investments in existing, widely deployed infrastructure.”	Paper, Tweet	
4) Graph of Thoughts - presents a prompting approach that models text generated by LLMs as an arbitrary graph; it enables combining arbitrary "thoughts" and enhancing them using feedback loops; the core idea is to enhance the LLM capabilities through "network reasoning" and without any model updates; this could be seen as a generalization of the now popular Chain-of-Thought and Tree-of-Thought.	Paper, Tweet	
5) MVDream - a multi-view diffusion model that can generate geometrically consistent multi-view images given a text prompt; it leverages pre-trained diffusion models and a multi-view dataset rendered from 3D assets; this leads to generalizability of 2D diffusion and consistency of 3D data.	Paper, Tweet	
6) Nougat - proposes an approach for neural optical understanding of academic documents; it supports the ability to extract text, equations, and tables from academic PDFs, i.e., convert PDFs into LaTeX/markdown.	Paper, Tweet	
7) Factuality Detection in LLMs - proposes a tool called FacTool to detect factual errors in texts generated by LLMs; shows the necessary components needed and the types of tools to integrate with LLMs for better detecting factual errors.	Paper, Tweet	
8) AnomalyGPT - an approach for industrial anomaly detection based on large vision-language models; it simulates anomalous images and textual descriptions to generate training data; employs an image decoder and prompt learner to detect anomalies; it shows few-shot in-context learning capabilities and achieves state-of-the-art performance benchmark datasets.	Paper, Tweet	
9) FaceChain - a personalized portrait generation framework combining customized image-generation models and face-related perceptual understanding models to generate truthful personalized portraits; it works with a handful of portrait images as input.	Paper	
10) Qwen-VL - introduces a set of large-scale vision-language models demonstrating strong performance in tasks like image captioning, question answering, visual localization, and flexible interaction.	Paper, Tweet	


Top ML Papers of the Week (August 21 - August 27)


Paper	Links	
1) Code Llama - a family of LLMs for code based on Llama 2; the models provided as part of this release: foundation base models	Paper, Tweet	
2) Survey on Instruction Tuning for LLMs - new survey paper on instruction tuning LLM, including a systematic review of the literature, methodologies, dataset construction, training models, applications, and more.	Paper, Tweet	
3) SeamlessM4T - a unified multilingual and multimodal machine translation system that supports ASR, text-to-text translation, speech-to-text translation, text-to-speech translation, and speech-to-speech translation.	Paper, Tweet	
4) Use of LLMs for Illicit Purposes - provides an overview of existing efforts to identify and mitigate threats and vulnerabilities arising from LLMs; serves as a guide to building more reliable and robust LLM-powered systems.	Paper, Tweet	
5) Giraffe - a new family of models that are fine-tuned from base Llama and Llama 2; extends the context length to 4K, 16K, and 32K; explores the space of expanding context lengths in LLMs so it also includes insights useful for practitioners and researchers.	Paper, Tweet	
6) IT3D - presents a strategy that leverages explicitly synthesized multi-view images to improve Text-to-3D generation; integrates a discriminator along a Diffusion-GAN dual training strategy to guide the training of the 3D models.	Paper	
7) A Survey on LLM-based Autonomous Agents - presents a comprehensive survey of LLM-based autonomous agents; delivers a systematic review of the field and a summary of various applications of LLM-based AI agents in domains like social science and engineering.	Paper, Tweet	
8) Prompt2Model - a new framework that accepts a prompt describing a task through natural language; it then uses the prompt to train a small special-purpose model that is conducive to deployment; the proposed pipeline automatically collects and synthesizes knowledge through three channels: dataset retrieval, dataset generation, and model retrieval.	Paper, Tweet	
9) LegalBench - a collaboratively constructed benchmark for measuring legal reasoning in LLMs; it consists of 162 tasks covering 6 different types of legal reasoning.	Paper, Tweet	
10) Language to Rewards for Robotic Skill Synthesis - proposes a new language-to-reward system that utilizes LLMs to define optimizable reward parameters to achieve a variety of robotic tasks; the method is evaluated on a real robot arm where complex manipulation skills such as non-prehensile pushing emerge.	Paper, Tweet	


Top ML Papers of the Week (August 14 - August 20)


Paper	Links	
1) Self-Alignment with Instruction Backtranslation - presents an approach to automatically label human-written text with corresponding instruction which enables building a high-quality instruction following language model; the steps are: 1) fine-tune an LLM with small seed data and web corpus, then 2) generate instructions for each web doc, 3) curate high-quality examples via the LLM, and finally 4) fine-tune on the newly curated data; the self-alignment approach outperforms all other Llama-based models on the Alpaca leaderboard.	Paper, Tweet	
2) Platypus - a family of fine-tuned and merged LLMs currently topping the Open LLM Leaderboard; it describes a process of efficiently fine-tuning and merging LoRA modules and also shows the benefits of collecting high-quality datasets for fine-tuning; specifically, it presents a small-scale, high-quality, and highly curated dataset, Open-Platypus, that enables strong performance with short and cheap fine-tuning time and cost... one can train a 13B model on a single A100 GPU using 25K questions in 5 hours.	Paper, Tweet	
3) Model Compression for LLMs - a short survey on the recent model compression techniques for LLMs; provides a high-level overview of topics such as quantization, pruning, knowledge distillation, and more; it also provides an overview of benchmark strategies and evaluation metrics for measuring the effectiveness of compressed LLMs.	Paper, Tweet	
4) GEARS - uses deep learning and gene relationship knowledge graph to help predict cellular responses to genetic perturbation; GEARS exhibited 40% higher precision than existing approaches in the task of predicting four distinct genetic interaction subtypes in a combinatorial perturbation screen.	Paper, Tweet	
5) Shepherd - introduces a language model (7B) specifically tuned to critique the model responses and suggest refinements; this enables the capability to identify diverse errors and suggest remedies; its critiques are either similar or preferred to ChatGPT.	Paper, Tweet	
6) Using GPT-4 Code Interpreter to Boost Mathematical Reasoning - proposes a zero-shot prompting technique for GPT-4 Code Interpreter that explicitly encourages the use of code for self-verification which further boosts performance on math reasoning problems; initial experiments show that GPT4-Code achieved a zero-shot accuracy of 69.7% on the MATH dataset which is an improvement of 27.5% over GPT-4’s performance (42.2%). Lots to explore here.	Paper, Tweet	
7) Teach LLMs to Personalize - proposes a general approach based on multitask learning for personalized text generation using LLMs; the goal is to have an LLM generate personalized text without relying on predefined attributes.	Paper, Tweet	
8) OctoPack - presents 4 terabytes of Git commits across 350 languages used to instruction tune code LLMs; achieves state-of-the-art performance among models not trained on OpenAI outputs, on the HumanEval Python benchmark; the data is also used to extend the HumanEval benchmark to other tasks such as code explanation and code repair.	Paper, Tweet	
9) Efficient Guided Generation for LLMs - presents a library to help LLM developers guide text generation in a fast and reliable way; provides generation methods that guarantee that the output will match a regular expression, or follow a JSON schema.	Paper, Tweet	
10) Bayesian Flow Networks - introduces a new class of generative models bringing together the power of Bayesian inference and deep learning; it differs from diffusion models in that it operates on the parameters of a data distribution rather than on a noisy version of the data; it’s adapted to continuous, discretized and discrete data with minimal changes to the training procedure.	Paper, Tweet	


Top ML Papers of the Week (August 7 - August 13)


Paper	Links	
1) LLMs as Database Administrators - presents D-Bot, a framework based on LLMs that continuously acquires database maintenance experience from textual sources; D-Bot can help in performing: 1) database maintenance knowledge detection from documents and tools, 2) tree of thought reasoning for root cause analysis, and 3) collaborative diagnosis among multiple LLMs.	Paper, Tweet	
2) Political Biases Found in NLP Models - develops methods to measure media biases in LLMs, including the fairness of downstream NLP models tuned on top of politically biased LLMs; findings reveal that LLMs have political leanings which reinforce existing polarization in the corpora.	Paper, Tweet	
3) Evaluating LLMs as Agents - presents a multidimensional benchmark (AgentBench) to assess LLM-as-Agent’s reasoning and decision-making abilities; results show that there is a significant disparity in performance between top commercial LLMs and open-source LLMs when testing the ability to act as agents; open-source LLMs lag on the AgentBench tasks while GPT-4 shows potential to build continuously learning agents.	Paper, Tweet	
4) Studying LLM Generalization with Influence Functions - introduces an efficient approach to scale influence functions to LLMs with up to 52 billion parameters; the influence functions are used to further investigate the generalization patterns of LLMs such as cross-lingual generalization and memorization; finds that middle layers in the network seem to be responsible for the most abstract generalization patterns.	Paper, Tweet	
5) Seeing Through the Brain - proposes NeuroImagen, a pipeline for reconstructing visual stimuli images from EEG signals to potentially understand visually-evoked brain activity; a latent diffusion model takes EEG data and reconstructs high-resolution visual stimuli images.	Paper, Tweet	
6) SynJax - is a new library that provides an efficient vectorized implementation of inference algorithms for structured distributions; it enables building large-scale differentiable models that explicitly model structure in data like tagging, segmentation, constituency trees, and spanning trees.	Paper, Tweet	
7) Synthetic Data Reduces Sycophancy in LLMs - proposes fine-tuning on simple synthetic data to reduce sycophancy in LLMs; sycophancy occurs when LLMs try to follow a user’s view even when it’s not objectively correct; essentially, the LLM repeats the user’s view even when the opinion is wrong.	Paper, Tweet	
8) Photorealistic Unreal Graphics (PUG) - presents photorealistic and semantically controllable synthetic datasets for representation learning using Unreal Engine; the goal is to democratize photorealistic synthetic data and enable more rigorous evaluations of vision models.	Paper, Tweet	
9) LLMs for Industrial Control - develops an approach to select demonstrations and generate high-performing prompts used with GPT for executing tasks such as controlling (Heating, Ventilation, and Air Conditioning) for buildings; GPT-4 performs comparable to RL method but uses fewer samples and lower technical debt.	Paper, Tweet	
10) Trustworthy LLMs - presents a comprehensive overview of important categories and subcategories crucial for assessing LLM trustworthiness; the dimensions include reliability, safety, fairness, resistance to misuse, explainability and reasoning, adherence to social norms, and robustness; finds that aligned models perform better in terms of trustworthiness but the effectiveness of alignment varies.	Paper, Tweet	


Top ML Papers of the Week (July 31 - August 6)


Paper	Links	
1) Open Problem and Limitation of RLHF - provides an overview of open problems and the limitations of RLHF.	Paper, Tweet	
2) Med-Flamingo - a new multimodal model that allows in-context learning and enables tasks such as few-shot medical visual question answering; evaluations based on physicians, show improvements of up to 20% in clinician's rating; the authors occasionally observed low-quality generations and hallucinations.	Paper, Tweet	
3) ToolLLM - enables LLMs to interact with 16000 real-world APIs; it’s a framework that allows data preparation, training, and evaluation; the authors claim that one of their models, ToolLLaMA, has reached the performance of ChatGPT (turbo-16k) in tool use.	Paper, Tweet	
4) Skeleton-of-Thought - proposes a prompting strategy that firsts generate an answer skeleton and then performs parallel API calls to generate the content of each skeleton point; reports quality improvements in addition to speed-up of up to 2.39x.	Paper, Tweet	
5) MetaGPT - a framework involving LLM-based multi-agents that encodes human standardized operating procedures (SOPs) to extend complex problem-solving capabilities that mimic efficient human workflows; this enables MetaGPT to perform multifaceted software development, code generation tasks, and even data analysis using tools like AutoGPT and LangChain.	Paper, Tweet	
6) OpenFlamingo - introduces a family of autoregressive vision-language models ranging from 3B to 9B parameters; the technical report describes the models, training data, and evaluation suite.	Paper, Tweet	
7) The Hydra Effect - shows that language models exhibit self-repairing properties — when one layer of attention heads is ablated it causes another later layer to take over its function.	Paper, Tweet	
8) Self-Check - explores whether LLMs have the capability to perform self-checks which is required for complex tasks that depend on non-linear thinking and multi-step reasoning; it proposes a zero-shot verification scheme to recognize errors without external resources; the scheme can improve question-answering performance through weighting voting and even improve math word problem-solving.	Paper, Tweet	
9) Agents Model the World with Language - presents an agent that learns a multimodal world model that predicts future text and image representations; it learns to predict future language, video, and rewards; it’s applied to different domains and can learn to follow instructions in visually and linguistically complex domains.	Paper, Tweet	
10) AutoRobotics-Zero - discovers zero-shot adaptable policies from scratch that enable adaptive behaviors necessary for sudden environmental changes; as an example, the authors demonstrate the automatic discovery of Python code for controlling a robot.	Paper, Tweet	


Top ML Papers of the Week (July 24 - July 30)


Paper	Links	
1) Universal Adversarial LLM Attacks - finds universal and transferable adversarial attacks that cause aligned models like ChatGPT and Bard to generate objectionable behaviors; the approach automatically produces adversarial suffixes using greedy and gradient search.	Paper, Tweet	
2) RT-2 - a new end-to-end vision-language-action model that learns from both web and robotics data; enables the model to translate the learned knowledge to generalized instructions for robotic control.	Paper, Tweet	
3) Med-PaLM Multimodal - introduces a new multimodal biomedical benchmark with 14 different tasks; it presents a proof of concept for a generalist biomedical AI system called Med-PaLM Multimodal; it supports different types of biomedical data like clinical text, imaging, and genomics.	Paper, Tweet	
4) Tracking Anything in High Quality - propose a framework for high-quality tracking anything in videos; consists of a video multi-object segmented and a pretrained mask refiner model to refine the tracking results; the model ranks 2nd place in the VOTS2023 challenge.	Paper, Tweet	
5) Foundation Models in Vision - presents a survey and outlook discussing open challenges and research directions for foundational models in computer vision.	Paper, Tweet	
6) L-Eval - a standardized evaluation for long context language models containing 411 long documents over 2K query-response pairs encompassing areas such as law, finance, school lectures, long conversations, novels, and meetings.	Paper, Tweet	
7) LoraHub - introduces LoraHub to enable efficient cross-task generalization via dynamic LoRA composition; it enables the combination of LoRA modules without human expertise or additional parameters/gradients; mimics the performance of in-context learning in few-shot scenarios.	Paper, Tweet	
8) Survey of Aligned LLMs - resents a comprehensive overview of alignment approaches, including aspects like data collection, training methodologies, and model evaluation.	Paper, Tweet	
9) WavJourney - leverages LLMs to connect various audio models to compose audio content for engaging storytelling; this involves an explainable and interactive design that enhances creative control in audio production.	Paper, Tweet	
10) FacTool - a task and domain agnostic framework for factuality detection of text generated by LLM; the effectiveness of the approach is tested on tasks such as code generation and mathematical reasoning; a benchmark dataset is released, including a ChatGPT plugin.	Paper, Tweet	


Top ML Papers of the Week (July 17 - July 23)


Paper	Links	
1) Llama 2 - a collection of pretrained foundational models and fine-tuned chat models ranging in scale from 7B to 70B; Llama 2-Chat is competitive on a range of tasks and shows strong results on safety and helpfulness.	Paper, Tweet	
2) How is ChatGPT’s Behavior Changing Over Time? - evaluates different versions of GPT-3.5 and GPT-4 on various tasks and finds that behavior and performance vary greatly over time; this includes differences in performance for tasks such as math problem-solving, safety-related generations, and code formatting.	Paper, Tweet	
3) FlashAttention-2 - improves work partitioning and parallelism and addresses issues like reducing non-matmul FLOPs, parallelizing attention computation which increases occupancy, and reducing communication through shared memory.	Paper, Tweet	
4) Measuring Faithfulness in Chain-of-Thought Reasoning - nds that CoT reasoning shows large variation across tasks by simple interventions like adding mistakes and paraphrasing; demonstrates that as the model becomes larger and more capable, the reasoning becomes less faithful; suggests carefully choosing the model size and tasks can enable CoT faithfulness.	Paper, Tweet	
5) Generative TV & Showrunner Agents - an approach to generate episodic content using LLMs and multi-agent simulation; this enables current systems to perform creative storytelling through the integration of simulation, the user, and powerful AI models and enhance the quality of AI-generated content.	Paper, Tweet	
6) Challenges & Application of LLMs - summarizes a comprehensive list of challenges when working with LLMs that range from brittle evaluations to prompt brittleness to a lack of robust experimental designs.	Paper, Tweet	
7) Retentive Network - presents a foundation architecture for LLMs with the goal to improve training efficiency, inference, and efficient long-sequence modeling; adapts retention mechanism for sequence modeling that support parallel representation, recurrent representations, and chunkwise recurrent representation.	Paper, Tweet	
8) Meta-Transformer - a framework that performs unified learning across 12 modalities; it can handle tasks that include fundamental perception (text, image, point cloud, audio, video), practical application (X-Ray, infrared, hyperspectral, and IMU), and data mining (graph, tabular, and time-series).	Paper, Tweet	
9) Retrieve In-Context Example for LLMs - presents a framework to iteratively train dense retrievers to identify high-quality in-context examples for LLMs; the approach enhances in-context learning performance demonstrated using a suite of 30 tasks; examples with similar patterns are helpful and gains are consistent across model sizes.	Paper, Tweet	
10) FLASK - proposes fine-grained evaluation for LLMs based on a range of alignment skill sets; involves 12 skills and can help to provide a holistic view of a model’s performance depending on skill, domain, and level of difficulty; useful to analyze factors that make LLMs more proficient at specific skills.	Paper, Tweet	


Top ML Papers of the Week (July 10 - July 16)


Paper	Links	
1) CM3Leon - introduces a retrieval-augmented multi-modal language model that can generate text and images; leverages diverse and large-scale instruction-style data for tuning which leads to significant performance improvements and 5x less training compute than comparable methods.	Paper, Tweet	
2) Claude 2 - presents a detailed model card for Claude 2 along with results on a range of safety, alignment, and capabilities evaluations.	Paper, Tweet	
3) Secrets of RLHF in LLMs - takes a closer look at RLHF and explores the inner workings of PPO with code included.	Paper, Tweet	
4) LongLLaMA - employs a contrastive training process to enhance the structure of the (key, value) space to extend context length; presents a fine-tuned model that lengthens context and demonstrates improvements in long context tasks.	Paper, Tweet	
5) Patch n’ Pack: NaViT - introduces a vision transformer for any aspect ratio and resolution through sequence packing; enables flexible model usage, improved training efficiency, and transfers to tasks involving image and video classification among others.	Paper, Tweet	
6) LLMs as General Pattern Machines - shows that even without any additional training, LLMs can serve as general sequence modelers, driven by in-context learning; this work applies zero-shot capabilities to robotics and shows that it’s possible to transfer the pattern among words to actions.	Paper, Tweet	
7) HyperDreamBooth - introduces a smaller, faster, and more efficient version of Dreambooth; enables personalization of text-to-image diffusion model using a single input image, 25x faster than Dreambooth.	Paper, Tweet	
8) Teaching Arithmetics to Small Transformers - trains small transformer models on chain-of-thought style data to significantly improve accuracy and convergence speed; it highlights the importance of high-quality instructive data for rapidly eliciting arithmetic capabilities.	Paper, Tweet	
9) AnimateDiff - appends a motion modeling module to a frozen text-to-image model, which is then trained and used to animate existing personalized models to produce diverse and personalized animated images.	Paper, Tweet	
10) Generative Pretraining in Multimodality - presents a new transformer-based multimodal foundation model to generate images and text in a multimodal context; enables performant multimodal assistants via instruction tuning.	Paper, Tweet	


Top ML Papers of the Week (July 3 - July 9)


Paper	Links	
1) A Survey on Evaluation of LLMs - a comprehensive overview of evaluation methods for LLMs focusing on what to evaluate, where to evaluate, and how to evaluate.	Paper, Tweet	
2) How Language Models Use Long Contexts - finds that LM performance is often highest when relevant information occurs at the beginning or end of the input context; performance degrades when relevant information is provided in the middle of a long context.	Paper, Tweet	
3) LLMs as Effective Text Rankers - proposes a prompting technique that enables open-source LLMs to perform state-of-the-art text ranking on standard benchmarks.	Paper, Tweet	
4) Multimodal Generation with Frozen LLMs - introduces an approach that effectively maps images to the token space of LLMs; enables models like PaLM and GPT-4 to tackle visual tasks without parameter updates; enables multimodal tasks and uses in-context learning to tackle various visual tasks.	Paper, Tweet	
5) CodeGen2.5 - releases a new code LLM trained on 1.5T tokens; the 7B model is on par with >15B code-generation models and it’s optimized for fast sampling.	Paper, Tweet	
6) Elastic Decision Transformer - introduces an advancement over Decision Transformers and variants by facilitating trajectory stitching during action inference at test time, achieved by adjusting to shorter history that allows transitions to diverse and better future states.	Paper, Tweet	
7) Robots That Ask for Help - presents a framework to measure and align the uncertainty of LLM-based planners that ask for help when needed.	Paper, Tweet	
8) Physics-based Motion Retargeting in Real-Time - proposes a method that uses reinforcement learning to train a policy to control characters in a physics simulator; it retargets motions in real-time from sparse human sensor data to characters of various morphologies.	Paper, Tweet	
9) Scaling Transformer to 1 Billion Tokens - presents LongNet, a Transformer variant that can scale sequence length to more than 1 billion tokens, with no loss in shorter sequences.	Paper, Tweet	
10) InterCode - introduces a framework of interactive coding as a reinforcement learning environment; this is different from the typical coding benchmarks that consider a static sequence-to-sequence process.	Paper, Tweet	


Top ML Papers of the Week (June 26 - July 2)


Paper	Links	
1) LeanDojo - an open-source Lean playground consisting of toolkits, data, models, and benchmarks for theorem proving; also develops ReProver, a retrieval augmented LLM-based prover for theorem solving using premises from a vast math library.	Paper, Tweet	
2) Extending Context Window of LLMs - extends the context window of LLMs like LLaMA to up to 32K with minimal fine-tuning (within 1000 steps); previous methods for extending the context window are inefficient but this approach attains good performance on several tasks while being more efficient and cost-effective.	Paper, Tweet	
3) Computer Vision Through the Lens of Natural Language - proposes a modular approach for solving computer vision problems by leveraging LLMs; the LLM is used to reason over outputs from independent and descriptive modules that provide extensive information about an image.	Paper, Tweet	
4) Visual Navigation Transformer - a foundational model that leverages the power of pretrained models to vision-based robotic navigation; it can be used with any navigation dataset and is built on a flexible Transformer-based architecture that can tackle various navigational tasks.	Paper, Tweet	
5) Generative AI for Programming Education - evaluates GPT-4 and ChatGPT on programming education scenarios and compares their performance with human tutors; GPT-4 outperforms ChatGPT and comes close to human tutors' performance.	Paper, Tweet	
6) DragDiffusion - extends interactive point-based image editing using diffusion models; it optimizes the diffusion latent to achieve precise spatial control and complete high-quality editing efficiently.	Paper, Tweet	
7) Understanding Theory-of-Mind in LLMs with LLMs - a framework for procedurally generating evaluations with LLMs; proposes a benchmark to study the social reasoning capabilities of LLMs with LLMs.	Paper, Tweet	
8) Evaluations with No Labels - a framework for self-supervised evaluation of LLMs by analyzing their sensitivity or invariance to transformations on input text; can be used to monitor LLM behavior on datasets streamed during live model deployment.	Paper, Tweet	
9) Long-range Language Modeling with Self-Retrieval - an architecture and training procedure for jointly training a retrieval-augmented language model from scratch for long-range language modeling tasks.	Paper, Tweet	
10) Scaling MLPs: A Tale of Inductive Bias - shows that the performance of MLPs improves with scale and highlights that lack of inductive bias can be compensated.	Paper, Tweet	


Top ML Papers of the Week (June 19 - June 25)


Paper	Links	
1) Textbooks Are All You Need - introduces a new 1.3B parameter LLM called phi-1; it’s significantly smaller in size and trained for 4 days using a selection of textbook-quality data and synthetic textbooks and exercises with GPT-3.5; achieves promising results on the HumanEval benchmark.	Paper, Tweet	
2) RoboCat - a new foundation agent that can operate different robotic arms and can solve tasks from as few as 100 demonstrations; the self-improving AI agent can self-generate new training data to improve its technique and get more efficient at adapting to new tasks.	Paper, Tweet	
3) ClinicalGPT - a language model optimized through extensive and diverse medical data, including medical records, domain-specific knowledge, and multi-round dialogue consultations.	Paper, Tweet	
4) An Overview of Catastrophic AI Risks - provides an overview of the main sources of catastrophic AI risks; the goal is to foster more understanding of these risks and ensure AI systems are developed in a safe manner.	Paper, Tweet	
5) LOMO - proposes a new memory-efficient optimizer that combines gradient computation and parameter update in one step; enables tuning the full parameters of an LLM with limited resources.	Paper, Tweet	
6) SequenceMatch - formulates sequence generation as an imitation learning problem; this framework allows the ability to incorporate backtracking into text generation through a backspace action; this enables the generative model to mitigate compounding errors by reverting sample tokens that lead to sequence OOD.	Paper, Tweet	
7) LMFlow - an extensible and lightweight toolkit that simplifies finetuning and inference of general large foundation models; supports continuous pretraining, instruction tuning, parameter-efficient finetuning, alignment tuning, and large model inference.	Paper, Tweet	
8) MotionGPT - uses multimodal control signals for generating consecutive human motions; it quantizes multimodal control signals intro discrete codes which are converted to LLM instructions that generate motion answers.	Paper, Tweet	
9) Wanda - introduces a simple and effective pruning approach for LLMs; it prunes weights with the smallest magnitudes multiplied by the corresponding input activations, on a per-output basis; the approach requires no retraining or weight update and outperforms baselines of magnitude pruning.	Paper, Tweet	
10) AudioPaLM - fuses text-based and speech-based LMs, PaLM-2 and AudioLM, into a multimodal architecture that supports speech understanding and generation; outperforms existing systems for speech translation tasks with zero-shot speech-to-text translation capabilities.	Paper, Tweet	


Top ML Papers of the Week (June 12 - June 18)


Paper	Links	
1) Voicebox - an all-in-one generative speech model; it can synthesize speech across 6 languages; it can perform noise removal, content editing, style conversion, and more; it's 20x faster than current models and outperforms single-purpose models through in-context learning.	Paper, Tweet	
2) FinGPT - an open-source LLM for the finance sector; it takes a data-centric approach, providing researchers & practitioners with accessible resources to develop FinLLMs.	Paper, Tweet	
3) Crowd Workers Widely Use Large Language Models for Text Production Tasks - estimates that 33-46% of crowd workers on MTurk used LLMs when completing a text production task.	Paper, Tweet	
4) Reliability of Watermarks for LLMs - watermarking is useful to detect LLM-generated text and potentially mitigate harms; this work studies the reliability of watermarking for LLMs and finds that watermarks are detectable even when the watermarked text is re-written by humans or paraphrased by another non-watermarked LLM.	Paper, Tweet	
5) Applications of Transformers - a new survey paper highlighting major applications of Transformers for deep learning tasks; includes a comprehensive list of Transformer models.	Paper, Tweet	
6) Benchmarking NN Training Algorithms - it’s currently challenging to properly assess the best optimizers to train neural networks; this paper presents a new benchmark, AlgoPerf, for benchmarking neural network training algorithms using realistic workloads.	Paper, Tweet	
7) Unifying LLMs & Knowledge Graphs - provides a roadmap for the unification of LLMs and KGs; covers how to incorporate KGs in LLM pre-training/inferencing, leverage LLMs for KG tasks such as question answering, and enhance both KGs and LLMs for bidirectional reasoning.	Paper, Tweet	
8) Augmenting LLMs with Long-term Memory - proposes a framework to enable LLMs to memorize long history; it’s enhanced with memory-augmented adaptation training to memorize long past context and use long-term memory for language modeling; achieves improvements on memory-augmented in-context learning over LLMs.	Paper, Tweet	
9) TAPIR - enables tracking any queried point on any physical surface throughout a video sequence; outperforms all baselines and facilitates fast inference on long and high-resolution videos (track points faster than real-time when using modern GPUs).	Paper, Tweet	
10) Mind2Web - a new dataset for evaluating generalist agents for the web; contains 2350 tasks from 137 websites over 31 domains; it enables testing generalization ability across tasks and environments, covering practical use cases on the web.	Paper, Tweet	


Top ML Papers of the Week (June 5 - June 11)


Paper	Links	
1) Tracking Everything Everywhere All at Once - propose a test-time optimization method for estimating dense and long-range motion; enables accurate, full-length motion estimation of every pixel in a video.	Paper, Tweet	
2) AlphaDev - a deep reinforcement learning agent which discovers faster sorting algorithms from scratch; the algorithms outperform previously known human benchmarks and have been integrated into the LLVM C++ library.	Paper, Tweet	
3) Sparse-Quantized Representation - a new compressed format and quantization technique that enables near-lossless compression of LLMs across model scales; “allows LLM inference at 4.75 bits with a 15% speedup”.	Paper, Tweet	
4) MusicGen - a simple and controllable model for music generation built on top of a single-stage transformer LM together with efficient token interleaving patterns; it can be conditioned on textual descriptions or melodic features and shows high performance on a standard text-to-music benchmark.	Paper, Tweet	
5) Augmenting LLMs with Databases - combines an LLM with a set of SQL databases, enabling a symbolic memory framework; completes tasks via LLM generating SQL instructions that manipulate the DB autonomously.	Paper, Tweet	
6) Concept Scrubbing in LLM - presents a method called LEAst-squares Concept Erasure (LEACE) to erase target concept information from every layer in a neural network; it’s used for reducing gender bias in BERT embeddings.	Paper , Tweet	
7) Fine-Grained RLHF - trains LMs with fine-grained human feedback; instead of using overall preference, more explicit feedback is provided at the segment level which helps to improve efficacy on long-form question answering, reduce toxicity, and enables LM customization.	Paper, Tweet	
8) Hierarchical Vision Transformer - pretrains vision transformers with a visual pretext task (MAE), while removing unnecessary components from a state-of-the-art multi-stage vision transformer; this enables a simple hierarchical vision transformer that’s more accurate and faster at inference and during training.	Paper, Tweet	
9) Humor in ChatGPT - explores ChatGPT’s capabilities to grasp and reproduce humor; finds that over 90% of 1008 generated jokes were the same 25 jokes and that ChatGPT is also overfitted to a particular joke structure.	Paper, Tweet	
10) Imitating Reasoning Process of Larger LLMs - develops a 13B parameter model that learns to imitate the reasoning process of large foundational models like GPT-4; it leverages large-scale and diverse imitation data and surpasses instruction-tuned models such as Vicuna-13B in zero-shot reasoning.	Paper, Tweet	


Top ML Papers of the Week (May 29-June 4)


Paper	Links	
1) Let’s Verify Step by Step - achieves state-of-the-art mathematical problem solving by rewarding each correct step of reasoning in a chain-of-thought instead of rewarding the final answer; the model solves 78% of problems from a representative subset of the MATH test set.	Paper, Tweet	
2) No Positional Encodings - shows that explicit position embeddings are not essential for decoder-only Transformers; shows that other positional encoding methods like ALiBi and Rotary are not well suited for length generalization.	Paper, Tweet	
3) BiomedGPT - a unified biomedical generative pretrained transformer model for vision, language, and multimodal tasks. Achieves state-of-the-art performance across 5 distinct tasks with 20 public datasets spanning over 15 unique biomedical modalities.	Paper, Tweet	
4) Thought Cloning - introduces an imitation learning framework to learn to think while acting; the idea is not only to clone the behaviors of human demonstrators but also the thoughts humans have when performing behaviors.	Paper, Tweet	
5) Fine-Tuning Language Models with Just Forward Passes - proposes a memory-efficient zeroth-order optimizer and a corresponding SGD algorithm to finetune large LMs with the same memory footprint as inference.	Paper , Tweet	
6) MERT - an acoustic music understanding model with large-scale self-supervised training; it incorporates a superior combination of teacher models to outperform conventional speech and audio approaches.	Paper , Tweet	
7) Bytes Are All You Need - investigates performing classification directly on file bytes, without needing to decode files at inference time; achieves ImageNet Top-1 accuracy of 77.33% using a transformer backbone; achieves 95.42% accuracy when operating on WAV files from the Speech Commands v2 dataset.	Paper, Tweet	
8) Direct Preference Optimization - while helpful to train safe and useful LLMs, the RLHF process can be complex and often unstable; this work proposes an approach to finetune LMs by solving a classification problem on the human preferences data, with no RL required.	Paper, Tweet	
9) SQL-PaLM - an LLM-based Text-to-SQL adopted from PaLM-2; achieves SoTA in both in-context learning and fine-tuning settings; the few-shot model outperforms the previous fine-tuned SoTA by 3.8% on the Spider benchmark; few-shot SQL-PaLM also outperforms few-shot GPT-4 by 9.9%, using a simple prompting approach.	Paper, Tweet	
10) CodeTF - an open-source Transformer library for state-of-the-art code LLMs; supports pretrained code LLMs and popular code benchmarks, including standard methods to train and serve code LLMs efficiently.	Paper, Tweet	


Top ML Papers of the Week (May 22-28)


Paper	Links	
1) QLoRA - an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU while preserving full 16-bit finetuning performance.	Paper, Tweet	
2) LIMA - a new 65B parameter LLaMa model fine-tuned on 1000 carefully curated prompts and responses; it doesn't use RLHF, generalizes well to unseen tasks not available in the training data, and generates responses equivalent or preferred to GPT-4 in 43% of cases, and even higher compared to Bard.	Paper, Tweet	
3) Voyager - an LLM-powered embodied lifelong learning agent in Minecraft that can continuously explore worlds, acquire skills, and make novel discoveries without human intervention.	Paper, Tweet	
4) Gorilla - a finetuned LLaMA-based model that surpasses GPT-4 on writing API calls. This capability can help identify the right API, boosting the ability of LLMs to interact with external tools to complete specific tasks.	Paper, Tweet	
5) The False Promise of Imitating Proprietary LLMs - provides a critical analysis of models that are finetuned on the outputs of a stronger model; argues that model imitation is a false premise and that the higher leverage action to improve open source models is to develop better base models.	Paper , Tweet	
6) Sophia - presents a simple scalable second-order optimizer that has negligible average per-step time and memory overhead; on language modeling, Sophia achieves 2x speed-up compared to Adam in the number of steps, total compute, and wall-clock time.	Paper , Tweet	
7) The Larger They Are, the Harder They Fail - shows that LLMs fail to generate correct Python code when default function names are swapped; they also strongly prefer incorrect continuation as they become bigger.	Paper, Tweet	
8) Model Evaluation for Extreme Risks - discusses the importance of model evaluation for addressing extreme risks and making responsible decisions about model training, deployment, and security.	Paper, Tweet	
9) LLM Research Directions - discusses a list of research directions for students looking to do research with LLMs.	Paper, Tweet	
10) Reinventing RNNs for the Transformer Era - proposes an approach that combines the efficient parallelizable training of Transformers with the efficient inference of RNNs; results show that the method performs on part with similarly sized Transformers.	Paper, Tweet	


Top ML Papers of the Week (May 15-21)


Paper	Links	
1) Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold - an approach for controlling GANs that allows dragging points of the image to precisely reach target points in a user-interactive manner.	Paper, Tweet	
2) Evidence of Meaning in Language Models Trained on Programs - argues that language models can learn meaning despite being trained only to perform next token prediction on text.	Paper, Tweet	
3) Towards Expert-Level Medical Question Answering with Large Language Models - a top-performing LLM for medical question answering; scored up to 86.5% on the MedQA dataset (a new state-of-the-art); approaches or exceeds SoTA across MedMCQA, PubMedQA, and MMLU clinical topics datasets.	Paper, Tweet	
4) MEGABYTE: Predicting Million-byte Sequences with Multiscale Transformers - a multi-scale decoder architecture enabling end-to-end modeling of sequences of over one million bytes; enables sub-quadratic self-attention and improved parallelism during decoding.	Paper, Tweet	
5) StructGPT: A General Framework for Large Language Model to Reason over Structured Data - improves the zero-shot reasoning ability of LLMs over structured data; effective for solving question answering tasks based on structured data.	Paper , Tweet	
6) TinyStories: How Small Can Language Models Be and Still Speak Coherent English? - uses a synthetic dataset of short stories to train and evaluate LMs that are much smaller than SoTA models but can produce fluent and consistent stories with several paragraphs, and demonstrate reasoning capabilities.	Paper , Tweet	
7) DoReMi: Optimizing Data Mixtures Speeds Up Language Model Pretraining - trains a small proxy model over domains to produce domain weights without knowledge of downstream tasks; it then resamples a dataset with the domain weights and trains a larger model; this enables using a 280M proxy model to train an 8B model (30x larger) more efficiently.	Paper, Tweet	
8) CodeT5+: Open Code Large Language Models for Code Understanding and Generation - supports a wide range of code understanding and generation tasks and different training methods to improve efficacy and computing efficiency; tested on 20 code-related benchmarks using different settings like zero-shot, fine-tuning, and instruction tuning; achieves SoTA on tasks like code completion, math programming, and text-to-code retrieval tasks.	Paper, Tweet	
9) Symbol tuning improves in-context learning in language models - an approach to finetune LMs on in-context input-label pairs where natural language labels are replaced by arbitrary symbols; boosts performance on unseen in-context learning tasks and algorithmic reasoning tasks.	Paper), Tweet	
10) Searching for Needles in a Haystack: On the Role of Incidental Bilingualism in PaLM's Translation Capability - shows that PaLM is exposed to over 30 million translation pairs across at least 44 languages; shows that incidental bilingualism connects to the translation capabilities of PaLM.	Paper, Tweet	


Top ML Papers of the Week (May 8-14)


Paper	Links	
1) LLM explains neurons in LLMs - applies GPT-4 to automatically write explanations on the behavior of neurons in LLMs and even score those explanations; this offers a promising way to improve interpretability in future LLMs and potentially detect alignment and safety problems.	Paper, Tweet	
2) PaLM 2 - a new state-of-the-art language model integrated into AI features and tools like Bard and the PaLM API; displays competitive performance in mathematical reasoning compared to GPT-4; instruction-tuned model, Flan-PaLM 2, shows good performance on benchmarks like MMLU and BIG-bench Hard.	Paper, Tweet	
3) ImageBind - an approach that learns joint embedding data across six modalities at once; extends zero-shot capabilities to new modalities and enables emergent applications including cross-modal retrieval, composing modalities with arithmetic, cross-modal detection, and generation.	Paper, Tweet	
4) TidyBot - shows that robots can combine language-based planning and perception with the few-shot summarization capabilities of LLMs to infer generalized user preferences that are applicable to future interactions.	Paper, Tweet	
5) Unfaithful Explanations in Chain-of-Thought Prompting - demonstrates that CoT explanations can misrepresent the true reason for a model’s prediction; when models are biased towards incorrect answers, CoT generation explanations supporting those answers.	Paper , Tweet	
6) InstructBLIP - explores visual-language instruction tuning based on the pre-trained BLIP-2 models; achieves state-of-the-art zero-shot performance on 13 held-out datasets, outperforming BLIP-2 and Flamingo.	Paper , Tweet	
7) Active Retrieval Augmented LLMs - introduces FLARE, retrieval augmented generation to improve the reliability of LLMs; FLARE actively decides when and what to retrieve across the course of the generation; demonstrates superior or competitive performance on long-form knowledge-intensive generation tasks.	Paper, Tweet	
8) FrugalGPT - presents strategies to reduce the inference cost associated with using LLMs while improving performance.	Paper, Tweet	
9) StarCoder - an open-access 15.5B parameter LLM with 8K context length and is trained on large amounts of code spanning 80+ programming languages.	Paper, Tweet	
10) MultiModal-GPT - a vision and language model for multi-round dialogue with humans; the model is fine-tuned from OpenFlamingo, with LoRA added in the cross-attention and self-attention parts of the language model.	Paper, Tweet	


Top ML Papers of the Week (May 1-7)


Paper	Links	
1) scGPT: Towards Building a Foundation Model for Single-Cell Multi-omics Using Generative AI - a foundation large language model pretrained on 10 million cells for single-cell biology.	Paper, Tweet	
2) GPTutor: a ChatGPT-powered programming tool for code explanation - a ChatGPT-powered tool for code explanation provided as a VSCode extension; claims to deliver more concise and accurate explanations than vanilla ChatGPT and Copilot; performance and personalization enhanced via prompt engineering; programmed to use more relevant code in its prompts.	Paper, Tweet	
3) Shap-E: Generating Conditional 3D Implicit Functions - a conditional generative model for 3D assets; unlike previous 3D generative models, this model generates implicit functions that enable rendering textured meshes and neural radiance fields.	Paper, Tweet	
4) Are Emergent Abilities of Large Language Models a Mirage? - presents an alternative explanation to the emergent abilities of LLMs; suggests that existing claims are creations of the researcher’s analyses and not fundamental changes in model behavior on specific tasks with scale	Paper, Tweet	
5) Interpretable Machine Learning for Science with PySR and SymbolicRegression.jl - releases PySR, an open-source library for practical symbolic regression for the sciences; it’s built on a high-performance distributed back-end and interfaces with several deep learning packages; in addition, a new benchmark, “EmpiricalBench”, is released to quantify applicability of symbolic regression algorithms in science.	Paper , Tweet	
6) PMC-LLaMA: Further Finetuning LLaMA on Medical Papers - a LLaMA model fine-tuned on 4.8 million medical papers; enhances capabilities in the medical domain and achieves high performance on biomedical QA benchmarks.	Paper , Tweet	
7) Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes - a mechanism to extract rationales from LLMs to train smaller models that outperform larger language models with less training data needed by finetuning or distillation.	Paper, Tweet	
8) Poisoning Language Models During Instruction Tuning - show that adversaries can poison LLMs during instruction tuning by contributing poison examples to datasets; it can induce degenerate outputs across different held-out tasks.	Paper, Tweet	
9) Unlimiformer: Long-Range Transformers with Unlimited Length Input - proposes long-range transformers with unlimited length input by augmenting pre-trained encoder-decoder transformer with external datastore to support unlimited length input; shows usefulness in long-document summarization; could potentially be used to improve the performance of retrieval-enhanced LLMs.	Paper, Tweet	
10) Learning to Reason and Memorize with Self-Notes - an approach that enables LLMs to reason and memorize enabling them to deviate from the input sequence at any time to explicitly “think”; this enables the LM to recall information and perform reasoning on the fly; experiments show that this method scales better to longer sequences unseen during training.	Paper, Tweet	


Top ML Papers of the Week (April 24 - April 30)


Paper	Links	
1) Learning Agile Soccer Skills for a Bipedal Robot with Deep Reinforcement Learning - applies deep reinforcement learning to synthesize agile soccer skills for a miniature humanoid robot; the resulting policy allows dynamic movement skills such as fast recovery, walking, and kicking.	Paper, Tweet	
2) Scaling Transformer to 1M tokens and beyond with RMT - leverages a recurrent memory transformer architecture to increase BERT’s effective context length to two million tokens while maintaining high memory retrieval accuracy.	Paper, Tweet	
3) Track Anything: Segment Anything Meets Videos - an interactive tool for video object tracking and segmentation; it’s built on top segment anything and allows flexible tracking and segmenting via user clicks.	Paper, Tweet	
4) A Cookbook of Self-Supervised Learning - provides an overview of fundamental techniques and key concepts in SSL; it also introduces practical considerations for implementing SSL methods successfully.	Paper, Tweet	
5) Harnessing the Power of LLMs in Practice: A Survey on ChatGPT and Beyond - a comprehensive and practical guide for practitioners working with LLMs; discusses many use cases with practical applications and limitations of LLMs in real-world scenarios.	Paper , Tweet	
6) AudioGPT: Understanding and Generating Speech, Music, Sound, and Talking Head - connects ChatGPT with audio foundational models to handle challenging audio tasks and a modality transformation interface to enable spoken dialogue.	Paper , Tweet	
7) DataComp: In search of the next generation of multimodal datasets - releases a new multimodal dataset benchmark containing 12.8B image-text pairs.	Paper, Tweet	
8) ChatGPT for Information Extraction - provides a deeper assessment of ChatGPT's performance on the important information extraction task.	Paper, Tweet	
9) Comparing Physician vs ChatGPT - investigates if chatbot assistants like ChatGPT can provide responses to patient questions while emphasizing quality and empathy; finds that chatbot responses were preferred over physician responses and rated significantly higher in terms of both quality and empathy.	Paper, Tweet	
10) Stable and low-precision training for large-scale vision-language models - introduces methods for accelerating and stabilizing training of large-scale language vision models.	Paper, Tweet	


Top ML Papers of the Week (April 17 - April 23)


Paper	Links	
1) DINOv2: Learning Robust Visual Features without Supervision - a new method for training high-performance computer vision models based on self-supervised learning; enables learning rich and robust visual features without supervision which are useful for both image-level visual tasks and pixel-level tasks; tasks supported include image classification, instance retrieval, video understanding, depth estimation, and much more.	Paper, Tweet	
2) Learning to Compress Prompts with Gist Tokens - an approach that trains language models to compress prompts into gist tokens reused for compute efficiency; this approach enables 26x compression of prompts, resulting in up to 40% FLOPs reductions.	Paper, Tweet	
3) Scaling the leading accuracy of deep equivariant models to biomolecular simulations of realistic size - presents a framework for large-scale biomolecular simulation; this is achieved through the high accuracy of equivariant deep learning and the ability to scale to large and long simulations; the system is able to “perform nanoseconds-long stable simulations of protein dynamics and scale up to a 44-million atom structure of a complete, all-atom, explicitly solvated HIV capsid on the Perlmutter supercomputer.”	Paper, Tweet	
4) Evaluating Verifiability in Generative Search Engines - performs human evaluation to audit popular generative search engines such as Bing Chat, Perplexity AI, and NeevaAI; finds that, on average, only 52% of generated sentences are supported by citations and 75% of citations support their associated sentence.	Paper, Tweet	
5) Generative Disco: Text-to-Video Generation for Music Visualization - an AI system based on LLMs and text-to-image models that generates music visualizations.	Paper , Tweet	
6) Architectures of Topological Deep Learning: A Survey on Topological Neural Networks	Paper , Tweet	
7) Visual Instruction Tuning - presents an approach that uses language-only GPT-4 to generate multimodal language-image instruction-following data; applies instruction tuning with the data and introduces LLaVA, an end-to-end trained large multimodal model for general-purpose visual and language understanding.	Paper, Tweet	
8) ChatGPT: Applications, Opportunities, and Threats	Paper, Tweet	
9) Chameleon: Plug-and-Play Compositional Reasoning with Large Language Models - a plug-and-play compositional reasoning framework that augments LLMs and can infer the appropriate sequence of tools to compose and execute in order to generate final responses; achieves 87% accuracy on ScienceQA and 99% on TabMWP.	Paper, Tweet	
10) Align your Latents: High-Resolution Video Synthesis with Latent Diffusion Models - applies latent diffusion models to high-resolution video generation; validates the model on creative content creation and real driving videos of 512 x 1024 and achieves state-of-the-art performance.	Paper, Tweet	


Top ML Papers of the Week (April 10 - April 16)


Paper	Links	
1) Zip-NeRF: Anti-Aliased Grid-Based Neural Radiance Fields - combines mip-NeRF 360 and grid-based models to improve NeRFs that train 22x faster than mip-NeRF 360.	Paper, Tweet	
2) Generative Agents: Interactive Simulacra of Human Behavior - proposes an architecture that extends LLMs to build agents that enable simulations of human-like behavior; these capabilities are possible by storing a complete record of an agent's experiences, synthesizing memories over time into higher-level reflections, and retrieving them dynamically to plan behavior.	Paper, Tweet	
3) Emergent autonomous scientific research capabilities of large language models - presents an agent that combines LLMs for autonomous design, planning, and execution of scientific experiments; shows emergent scientific research capabilities, including the successful performance of catalyzed cross-coupling reactions.	Paper, Tweet	
4) Automatic Gradient Descent: Deep Learning without Hyperparameters - derives optimization algorithms that explicitly leverage neural architecture; it proposes a first-order optimizer without hyperparameters that trains CNNs at ImageNet scale.	Paper, Tweet	
5) ChemCrow: Augmenting large-language models with chemistry tools - presents an LLM chemistry agent that performs tasks across synthesis, drug discovery, and materials design; it integrates 13 expert-design tools to augment LLM performance in chemistry and demonstrate effectiveness in automating chemical tasks.	Paper , Tweet	
6) One Small Step for Generative AI, One Giant Leap for AGI: A Complete Survey on ChatGPT in AIGC Era - A Survey of ChatGPT and GPT-4	Paper , Tweet	
7) OpenAGI: When LLM Meets Domain Experts - an open-source research platform to facilitate the development and evaluation of LLMs in solving complex, multi-step tasks through manipulating various domain expert models.	Paper, Tweet	
8) AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models - a new benchmark to assess foundational models in the context of human-centric standardized exams, including college entrance exams, law school admission tests, and math competitions, among others.	Paper, Tweet	
9) Teaching Large Language Models to Self-Debug - proposes an approach that teaches LLMs to debug their predicted program via few-shot demonstrations; this allows a model to identify its mistakes by explaining generated code in natural language; achieves SoTA on several code generation tasks like text-to-SQL generation.	Paper, Tweet	
10) Segment Everything Everywhere All at Once - a promptable, interactive model for various segmentation tasks that yields competitive performance on open-vocabulary and interactive segmentation benchmarks.	Paper, Tweet	
Top ML Papers of the Week (April 3 - April 9)


Paper	Links	
1) Segment Anything - presents a set of resources to establish foundational models for image segmentation; releases the largest segmentation dataset with over 1 billion masks on 11M licensed images; the model’s zero-shot performance is competitive with or even superior to fully supervised results.	Paper, Tweet	
2) Instruction Tuning with GPT-4 - presents GPT-4-LLM, a "first attempt" to use GPT-4 to generate instruction-following data for LLM fine-tuning; the dataset is released and includes 52K unique English and Chinese instruction-following data; the dataset is used to instruction-tune LLaMA models which leads to superior zero-shot performance on new tasks.	Paper, Tweet	
3) Eight Things to Know about Large Language Models - discusses important considerations regarding the capabilities and limitations of LLMs.	Paper, Tweet	
4) A Survey of Large Language Models - a new 50 pages survey on large language models.	Paper, Tweet	
5) Baize: An Open-Source Chat Model with Parameter-Efficient Tuning on Self-Chat Data - an open-source chat model fine-tuned with LoRA. Leverages 100K dialogs generated from ChatGPT chatting with itself; it releases the dialogs along with 7B, 13B, and 30B parameter models.	Paper , Tweet	
6) Do the Rewards Justify the Means? Measuring Trade-Offs Between Rewards and Ethical Behavior in the MACHIAVELLI Benchmark - a new benchmark of 134 text-based Choose-Your-Own-Adventure games to evaluate the capabilities and unethical behaviors of LLMs.	Paper , Tweet	
7) Better Language Models of Code through Self-Improvement - generates pseudo data from knowledge gained through pre-training and fine-tuning; adds the data to the training dataset for the next step; results show that different frameworks can be improved in performance using code-related generation tasks.	Paper, Tweet	
8) Summary of ChatGPT/GPT-4 Research and Perspective Towards the Future of Large Language Models - an overview of applications of ChatGPT and GPT-4; the analysis is done on 194 relevant papers and discusses capabilities, limitations, concerns, and more	Paper, Tweet	
9) Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling - a suite for analyzing LLMs across training and scaling; includes 16 LLMs trained on public data and ranging in size from 70M to 12B parameters.	Paper, Tweet	
10) SegGPT: Segmenting Everything In Context - unifies segmentation tasks into a generalist model through an in-context framework that supports different kinds of data.	Paper, Tweet	


Top ML Papers of the Week (Mar 27 - April 2)


Paper	Links	
1) BloombergGPT: A Large Language Model for Finance - a new 50B parameter large language model for finance. Claims the largest domain-specific dataset yet with 363 billion tokens... further augmented with 345 billion tokens from general-purpose datasets; outperforms existing models on financial tasks while not sacrificing performance on general LLM benchmarks.	Paper, Tweet	
2) Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware - a low-cost system that performs end-to-end imitation learning from real demonstrations; also presents an algorithm called Action Chunking with Transformers to learn a generative model that allows a robot to learn difficult tasks in the real world.	Paper, Tweet	
3) HuggingGPT: Solving AI Tasks with ChatGPT and its Friends in HuggingFace - a system that leverages LLMs like ChatGPT to conduct task planning, select models and act as a controller to execute subtasks and summarize responses according to execution results.	Paper, Tweet	
4) ChatDoctor: A Medical Chat Model Fine-tuned on LLaMA Model using Medical Domain Knowledge - a medical chat model fine-tuned on LLaMA using medical domain knowledge. Collects data on around 700 diseases and generated 5K doctor-patient conversations to finetune the LLM.	Paper, Tweet	
5) LLaMA-Adapter: Efficient Fine-tuning of Language Models with Zero-init Attention - a lightweight adaption method to efficiently fine-tune LLaMA into an instruction-following model; generates responses comparable to Alpaca with fully fine-tuned 7B parameter; it’s also extended for multi-modal input support.	Paper , Tweet	
6) ChatGPT Outperforms Crowd-Workers for Text-Annotation Tasks - demonstrates that ChatGPT can outperform crowd-workers for several annotation tasks such as relevance, topics, and frames detection; besides better zero-shot accuracy, the per-annotation cost of ChatGPT is less 20 times cheaper than MTurk.	Paper , Tweet	
7) Language Models can Solve Computer Tasks - shows that a pre-trained LLM agent can execute computer tasks using a simple prompting scheme where the agent recursively criticizes and improves its outputs.	Paper, Tweet	
8) DERA: Enhancing Large Language Model Completions with Dialog-Enabled Resolving Agents - a paradigm to enhance large language model completions by allowing models to communicate feedback and iteratively improve output; DERA outperforms base GPT-4 on clinically-focused tasks.	Paper, Tweet	
9) Natural Selection Favors AIs over Humans - discusses why AI systems will become more fit than humans and the potential dangers and risks involved, including ways to mitigate them.	Paper, Tweet	
10) Machine Learning for Partial Differential Equations - Pa review examining avenues of partial differential equations research advanced by machine learning.	Paper, Tweet	


Top ML Papers of the Week (Mar 20-Mar 26)


Paper	Links	
1) Sparks of Artificial General Intelligence: Early experiments with GPT-4 - a comprehensive investigation of an early version of GPT-4 when it was still in active development by OpenAI.	Paper, Tweet	
2) Reflexion: an autonomous agent with dynamic memory and self-reflection - proposes an agent with dynamic memory and self-reflection capabilities to enhance its existing reasoning trace and task-specific action choice abilities.	Paper, Tweet	
3) Capabilities of GPT-4 on Medical Challenge Problems - shows that GPT-4 exceeds the passing score on USMLE by over 20 points and outperforms GPT-3.5 as well as models specifically fine-tuned on medical knowledge (Med-PaLM, a prompt-tuned version of Flan-PaLM 540B).	Paper, Tweet	
4) GPTs are GPTs: An Early Look at the Labor Market Impact Potential of Large Language Models - investigates the potential implications of GPT models and related systems on the US labor market.	Paper, Tweet	
5) CoLT5: Faster Long-Range Transformers with Conditional Computation - a long-input Transformer model that employs conditional computation, devoting more resources to important tokens in both feedforward and attention layers.	Paper , Tweet	
6) Artificial muses: Generative Artificial Intelligence Chatbots Have Risen to Human-Level Creativity - compares human-generated ideas with those generated by generative AI chatbots like ChatGPT and YouChat; reports that 9.4% of humans were more creative than GPT-4 and that GAIs are valuable assistants in the creative process.	Paper , Tweet	
7) A Comprehensive Capability Analysis of GPT-3 and GPT-3.5 Series Models - a comprehensive capability analysis of GPT series models; evaluates performance on 9 natural language understanding tasks using 21 datasets.	Paper, Tweet	
8) Context-faithful Prompting for Large Language Models - presents a prompting technique that aims to improve LLMs' faithfulness using strategies such as opinion-based prompts and counterfactual demonstrations.	Paper, Tweet	
9) Text2Room: Extracting Textured 3D Meshes from 2D Text-to-Image Models - a method for extracting room-scale textured 3D meshes from 2D text-to-image models.	Paper, ProjectTweet	
10) PanGu-Σ: Towards Trillion Parameter Language Model with Sparse Heterogeneous Computing - a trillion parameter language model with sparse heterogeneous computing.	Paper, Tweet	


Top ML Papers of the Week (Mar 13-Mar 19)


Paper	Links	
1) GPT-4 Technical Report - GPT-4 - a large multimodal model with broader general knowledge and problem-solving abilities.	Paper, Tweet	
2) LERF: Language Embedded Radiance Fields - a method for grounding language embeddings from models like CLIP into NeRF; this enables open-ended language queries in 3D.	Paper, Tweet	
3) An Overview on Language Models: Recent Developments and Outlook - an overview of language models covering recent developments and future directions. It also covers topics like linguistic units, structures, training methods, evaluation, and applications.	Paper, Tweet	
4) Eliciting Latent Predictions from Transformers with the Tuned Lens - a method for transformer interpretability that can trace a language model predictions as it develops layer by layer.	Paper, Tweet	
5) Meet in the Middle: A New Pre-training Paradigm - a new pre-training paradigm using techniques that jointly improve training data efficiency and capabilities of LMs in the infilling task; performance improvement is shown in code generation tasks.	Paper , Tweet	
6) Resurrecting Recurrent Neural Networks for Long Sequences - demonstrates that careful design of deep RNNs using standard signal propagation arguments can recover the performance of deep state-space models on long-range reasoning tasks.	Paper , Tweet	
7) UPRISE: Universal Prompt Retrieval for Improving Zero-Shot Evaluation - a new approach to tune a lightweight and versatile retriever to automatically retrieve prompts to improve zero-shot performance and help mitigate hallucinations.	Paper, Tweet	
8) Patches Are All You Need? - proposes ConvMixer, a parameter-efficient fully-convolutional model which replaces self-attention and MLP layers in ViTs with less-expressive depthwise and pointwise convolutional layers.	Paper, Tweet	
9) NeRFMeshing: Distilling Neural Radiance Fields into Geometrically-Accurate 3D Meshes - a compact and flexible architecture that enables easy 3D surface reconstruction from any NeRF-driven approach; distills NeRFs into geometrically-accurate 3D meshes.	Paper, Tweet	
10) High-throughput Generative Inference of Large Language Models with a Single GPU - a high-throughput generation engine for running LLMs with limited GPU memory.	Paper, Code , Tweet	


Top ML Papers of the Week (Mar 6-Mar 12)


Paper	Links	
1) PaLM-E: An Embodied Multimodal Language Model - incorporates real-world continuous sensor modalities resulting in an embodied LM that performs tasks such as robotic manipulation planning, visual QA, and other embodied reasoning tasks.	Paper, Demo , Tweet	
2) Prismer: A Vision-Language Model with An Ensemble of Experts - a parameter-efficient vision-language model powered by an ensemble of domain experts; it efficiently pools expert knowledge from different domains and adapts it to various vision-language reasoning tasks.	Paper, GitHub, Project , Tweet	
3) Visual ChatGPT: Talking, Drawing and Editing with Visual Foundation Models - it connects ChatGPT and different visual foundation models to enable users to interact with ChatGPT beyond language format.	Paper, GitHub Tweet	
4) A Comprehensive Survey of AI-Generated Content (AIGC): A History of Generative AI from GAN to ChatGPT - an overview of generative AI - from GAN to ChatGPT.	Paper, Tweet	
5) Larger language models do in-context learning differently - shows that with scale, LLMs can override semantic priors when presented with enough flipped labels; these models can also perform well when replacing targets with semantically-unrelated targets.	Paper , Tweet	
6) Foundation Models for Decision Making: Problems, Methods, and Opportunities - provides an overview of foundation models for decision making, including tools, methods, and new research directions.	Project , Tweet	
7) Hyena Hierarchy: Towards Larger Convolutional Language Models - a subquadratic drop-in replacement for attention; it interleaves implicit long convolutions and data-controlled gating and can learn on sequences 10x longer and up to 100x faster than optimized attention.	Paper, Code, Blog, Tweet	
8) OpenICL: An Open-Source Framework for In-context Learning - a new open-source toolkit for in-context learning and LLM evaluation; supports various state-of-the-art retrieval and inference methods, tasks, and zero-/few-shot evaluation of LLMs.	Paper, Repo, Tweet	
9) MathPrompter: Mathematical Reasoning using Large Language Models - a technique that improves LLM performance on mathematical reasoning problems; it uses zero-shot chain-of-thought prompting and verification to ensure generated answers are accurate.	Paper, Tweet	
10) Scaling up GANs for Text-to-Image Synthesis - enables scaling up GANs on large datasets for text-to-image synthesis; it’s found to be orders of magnitude faster at inference time, synthesizes high-resolution images, & supports various latent space editing applications.	Paper, Project , Tweet	


Top ML Papers of the Week (Feb 27-Mar 5)


Paper	Links	
1) Language Is Not All You Need: Aligning Perception with Language Models - introduces a multimodal large language model called Kosmos-1; achieves great performance on language understanding, OCR-free NLP, perception-language tasks, visual QA, and more.	Paper, Tweet	
2) Evidence of a predictive coding hierarchy in the human brain listening to speech - finds that human brain activity is best explained by the activations of modern language models enhanced with long-range and hierarchical predictions.	Paper, Tweet	
3) EvoPrompting: Language Models for Code-Level Neural Architecture Search - combines evolutionary prompt engineering with soft prompt-tuning to find high-performing models; it leverages few-shot prompting which is further improved by using an evolutionary search approach to improve the in-context examples.	Paper, Tweet	
4) Consistency Models - a new family of generative models that achieve high sample quality without adversarial training.	Paper, Tweet	
5) Goal Driven Discovery of Distributional Differences via Language Descriptions - a new task that automatically discovers corpus-level differences via language description in a goal-driven way; applications include discovering insights from commercial reviews and error patterns in NLP systems.	Paper , Code, Tweet	
6) High-resolution image reconstruction with latent diffusion models from human brain activity - proposes an approach for high-resolution image reconstruction with latent diffusion models from human brain activity.	Project , Tweet	
7) Grounded Decoding: Guiding Text Generation with Grounded Models for Robot Control - a scalable approach to planning with LLMs in embodied settings through grounding functions; GD is found to be a general, flexible, and expressive approach to embodied tasks.	Paper, Project Tweet	
8) Language-Driven Representation Learning for Robotics - a framework for language-driven representation learning from human videos and captions for robotics.	Paper, Models, Evaluation, Tweet	
9) Dropout Reduces Underfitting - demonstrates that dropout can mitigate underfitting when used at the start of training; it counteracts SGD stochasticity and limits the influence of individual batches when training models.	Paper, Tweet	
10) Enabling Conversational Interaction with Mobile UI using Large Language Models - an approach that enables versatile conversational interactions with mobile UIs using a single LLM.	Paper, Tweet	


Top ML Papers of the Week (Feb 20-26)


Paper	Links	
1) LLaMA: Open and Efficient Foundation Language Models - a 65B parameter foundation model released by Meta AI; relies on publicly available data and outperforms GPT-3 on most benchmarks despite being 10x smaller.	Paper, Tweet	
2) Composer: Creative and Controllable Image Synthesis with Composable Conditions - a 5B parameter creative and controllable diffusion model trained on billions (text, image) pairs.	Paper, Project , GitHub , Tweet	
3) The Wisdom of Hindsight Makes Language Models Better Instruction Followers - an alternative algorithm to train LLMs from feedback; the feedback is converted to instruction by relabeling the original one and training the model, in a supervised way, for better alignment.	Paper, GitHub Tweet	
4) Active Prompting with Chain-of-Thought for Large Language Models - a prompting technique to adapt LLMs to different task-specific example prompts (annotated with human-designed chain-of-thought reasoning); this process involves finding where the LLM is most uncertain and annotating those.	Paper, Code Tweet	
5) Modular Deep Learning - a survey offering a unified view of the building blocks of modular neural networks; it also includes a discussion about modularity in the context of scaling LMs, causal inference, and other key topics in ML.	Paper , Project, Tweet	
6) Recitation-Augmented Language Models - an approach that recites passages from the LLM’s own memory to produce final answers; shows high performance on knowledge-intensive tasks.	Paper , Tweet	
7) Learning Performance-Improving Code Edits - an approach that uses LLMs to suggest functionally correct, performance-improving code edits.	Paper, Tweet	
8) More than you've asked for: A Comprehensive Analysis of Novel Prompt Injection Threats to Application-Integrated Large Language Models - a comprehensive analysis of novel prompt injection threats to application-integrated LLMs.	Paper, Tweet	
9) Aligning Text-to-Image Models using Human Feedback - proposes a fine-tuning method to align generative models using human feedback.	Paper, Tweet	
10) MERF: Memory-Efficient Radiance Fields for Real-time View Synthesis in Unbounded Scenes - a memory-efficient radiance field representation for real-time view synthesis of large-scale scenes in a browser.	Paper, Tweet	


Top ML Papers of the Week (Feb 13 - 19)


Paper	Links	
1) Symbolic Discovery of Optimization Algorithms - a simple and effective optimization algorithm that’s more memory-efficient than Adam.	Paper, Tweet	
2) Transformer models: an introduction and catalog	Paper, Tweet	
3) 3D-aware Conditional Image Synthesis - a 3D-aware conditional generative model extended with neural radiance fields for controllable photorealistic image synthesis.	Project Tweet	
4) The Capacity for Moral Self-Correction in Large Language Models - finds strong evidence that language models trained with RLHF have the capacity for moral self-correction. The capability emerges at 22B model parameters and typically improves with scale.	Paper, Tweet	
5) Vision meets RL - uses reinforcement learning to align computer vision models with task rewards; observes large performance boost across multiple CV tasks such as object detection and colorization.	Paper	
6) Language Quantized AutoEncoders: Towards Unsupervised Text-Image Alignment - an unsupervised method for text-image alignment that leverages pretrained language models; it enables few-shot image classification with LLMs.	Paper , Code Tweet	
7) Augmented Language Models: a Survey - a survey of language models that are augmented with reasoning skills and the capability to use tools.	Paper, Tweet	
8) Geometric Clifford Algebra Networks - an approach to incorporate geometry-guided transformations into neural networks using geometric algebra.	Paper, Tweet	
9) Auditing large language models: a three-layered approach - proposes a policy framework for auditing LLMs.	Paper, Tweet	
10) Energy Transformer - a transformer architecture that replaces the sequence of feedforward transformer blocks with a single large Associate Memory model; this follows the popularity that Hopfield Networks have gained in the field of ML.	Paper, Tweet	


Top ML Papers of the Week (Feb 6 - 12)


Paper	Links	
1) Toolformer: Language Models Can Teach Themselves to Use Tools - introduces language models that teach themselves to use external tools via simple API calls.	Paper, Tweet	
2) Describe, Explain, Plan and Select: Interactive Planning with Large Language Models Enables Open-World Multi-Task Agents - proposes using language models for open-world game playing.	Paper, Tweet	
3) A Categorical Archive of ChatGPT Failures - a comprehensive analysis of ChatGPT failures for categories like reasoning, factual errors, maths, and coding.	Paper, Tweet	
4) Hard Prompts Made Easy: Gradient-Based Discrete Optimization for Prompt Tuning and Discovery - optimizing hard text prompts through efficient gradient-based optimization.	Paper, Tweet	
5) Data Selection for Language Models via Importance Resampling - proposes a cheap and scalable data selection framework based on an importance resampling algorithm to improve the downstream performance of LMs.	Paper, Tweet	
6) Structure and Content-Guided Video Synthesis with Diffusion Models - proposes an approach for structure and content-guided video synthesis with diffusion models.	Paper , Project, Tweet	
7) A Multitask, Multilingual, Multimodal Evaluation of ChatGPT on Reasoning, Hallucination, and Interactivity - performs a more rigorous evaluation of ChatGPt on reasoning, hallucination, and interactivity.	Paper, Tweet	
8) Noise2Music: Text-conditioned Music Generation with Diffusion Models - proposes diffusion models to generate high-quality 30-second music clips via text prompts.	Paper, Project, Tweet	
9) Offsite-Tuning: Transfer Learning without Full Model - introduces an efficient, privacy-preserving transfer learning framework to adapt foundational models to downstream data without access to the full model.	Paper, Project, Tweet	
10) Zero-shot Image-to-Image Translation - proposes a model for zero-shot image-to-image translation.	Paper, Project, Tweet	


Top ML Papers of the Week (Jan 30-Feb 5)


Paper	Links	
1) REPLUG: Retrieval-Augmented Black-Box Language Models - a retrieval-augmented LM framework that adapts a retriever to a large-scale, black-box LM like GPT-3.	Paper, Tweet	
2) Extracting Training Data from Diffusion Models - shows that diffusion-based generative models can memorize images from the training data and emit them at generation time.	Paper, Tweet	
3) The Flan Collection: Designing Data and Methods for Effective Instruction Tuning - release a more extensive publicly available collection of tasks, templates, and methods to advancing instruction-tuned models.	Paper, Tweet	
4) Multimodal Chain-of-Thought Reasoning in Language Models - incorporates vision features to elicit chain-of-thought reasoning in multimodality, enabling the model to generate effective rationales that contribute to answer inference.	Paper, Code Tweet	
5) Dreamix: Video Diffusion Models are General Video Editors - a diffusion model that performs text-based motion and appearance editing of general videos.	Paper, Project, Tweet	
6) Benchmarking Large Language Models for News Summarization	Paper , Tweet	
7) Mathematical Capabilities of ChatGPT - investigates the mathematical capabilities of ChatGPT on a new holistic benchmark called GHOSTS.	Paper, Tweet	
8) Emergence of Maps in the Memories of Blind Navigation Agents - trains an AI agent to navigate purely by feeling its way around; no use of vision, audio, or any other sensing (as in animals).	Paper, Project, Tweet	
9) SceneDreamer: Unbounded 3D Scene Generation from 2D Image Collections - a generative model that synthesizes large-scale 3D landscapes from random noises.	Paper, Tweet	
10) Large Language Models Can Be Easily Distracted by Irrelevant Context - finds that many prompting techniques fail when presented with irrelevant context for arithmetic reasoning.	Paper, Tweet	


Top ML Papers of the Week (Jan 23-29)


Paper	Links	
1) MusicLM: Generating Music From Text - a generative model for generating high-fidelity music from text descriptions.	Paper, Tweet	
2) Hungry Hungry Hippos: Towards Language Modeling with State Space Models - an approach to reduce the gap, in terms of performance and hardware utilization, between state space models and attention for language modeling.	Paper, Tweet	
3) A Watermark for Large Language Models - a watermarking framework for proprietary language models.	Paper, Tweet	
4) Text-To-4D Dynamic Scene Generation - a new text-to-4D model for dynamic scene generation from input text.	Paper, GitHub, Tweet	
5) ClimaX: A foundation model for weather and climate - a foundation model for weather and climate, including many capabilities for atmospheric science tasks.	Paper, Tweet, Blog	
6) Open Problems in Applied Deep Learning - If you're looking for interesting open problems in DL, this is a good reference. Not sure if intentional but it also looks useful to get a general picture of current trends in deep learning with ~300 references.	Paper , Tweet	
7) DetectGPT: Zero-Shot Machine-Generated Text Detection using Probability Curvature - an approach for zero-shot machine-generated text detection. Uses raw log probabilities from the LLM to determine if the passage was sampled from it.	Paper, Tweet	
8) StyleGAN-T: Unlocking the Power of GANs for Fast Large-Scale Text-to-Image Synthesis - a new model that aims to regain the competitiveness of GANs for fast large-scale text-to-image synthesis.	Paper, Project, Code Tweet	
9) Large language models generate functional protein sequences across diverse families - an LLM that can generate protein sequences with a predictable function across large protein families.	Paper, Tweet	
10) The Impossibility of Parallelizing Boosting - investigates the possibility of parallelizing boosting.	Paper, Tweet	


Top ML Papers of the Week (Jan 16-22)


Paper	Links	
1) Google AI Research Recap (2022 Edition) - an excellent summary of some notable research Google AI did in 2022.	Blog, Tweet	
2) Dissociating language and thought in large language models: a cognitive perspective - a review paper on the capabilities of LLMs from a cognitive science perspective.	Paper, Tweet	
3) Human-Timescale Adaptation in an Open-Ended Task Space - an agent trained at scale that leads to a general in-content learning algorithm able to adapt to open-ended embodied 3D problems.	Paper, Tweet	
4) AtMan: Understanding Transformer Predictions Through Memory Efficient Attention Manipulation - an approach to help provide explanations of generative transformer models through memory-efficient attention manipulation.	Paper, Tweet	
5) Everything is Connected: Graph Neural Networks - short overview of key concepts in graph representation learning.	Paper, Tweet	
6) GLIGEN: Open-Set Grounded Text-to-Image Generation - an approach that extends the functionality of existing pre-trained text-to-image diffusion models by enabling conditioning on grounding inputs.	Paper, Tweet, Project	
7) InstructPix2Pix: Learning to Follow Image Editing Instructions - proposes a method with the capability of editing images from human instructions.	Paper, Tweet	
8) Dataset Distillation: A Comprehensive Review	Paper, Tweet	
9) Learning-Rate-Free Learning by D-Adaptation - a new method for automatically adjusting the learning rate during training, applicable to more than a dozen diverse ML problems.	Paper, Tweet	
10) RecolorNeRF: Layer Decomposed Radiance Field for Efficient Color Editing of 3D Scenes - a user-friendly color editing approach for the neural radiance field to achieve a more efficient view-consistent recoloring.	Paper, Tweet	


Top ML Papers of the Week (Jan 9-15)


Paper	Links	
1) Mastering Diverse Domains through World Models - a general algorithm to collect diamonds in Minecraft from scratch without human data or curricula, a long-standing challenge in AI.	Paper, Tweet	
2) Tracr: Compiled Transformers as a Laboratory for Interpretability - a compiler for converting RASP programs into transformer weights. This way of constructing NNs weights enables the development and evaluation of new interpretability tools.	Paper, Tweet, Code	
3) Multimodal Deep Learning - multimodal deep learning is a new book published on ArXiv.	Book, Tweet	
4) Forecasting Potential Misuses of Language Models for Disinformation Campaigns—and How to Reduce Risk - new work analyzing how generative LMs could potentially be misused for disinformation and how to mitigate these types of risks.	Paper, Tweet	
5) Why do Nearest Neighbor Language Models Work? - empirically identifies reasons why retrieval-augmented LMs (specifically k-nearest neighbor LMs) perform better than standard parametric LMs.	Paper, Code, Tweet	
6) Memory Augmented Large Language Models are Computationally Universal - investigates the use of existing LMs (e.g, Flan-U-PaLM 540B) combined with associative read-write memory to simulate the execution of a universal Turing machine.	Paper , Tweet	
7) A Survey on Transformers in Reinforcement Learning - transformers for RL will be a fascinating research area to track. The same is true for the reverse direction (RL for Transformers)... a notable example: using RLHF to improve LLMs (e.g., ChatGPT).	Paper, Tweet	
8) Scaling Laws for Generative Mixed-Modal Language Models - introduces scaling laws for generative mixed-modal language models.	Paper, Tweet	
9) DeepMatcher: A Deep Transformer-based Network for Robust and Accurate Local Feature Matching - a transformer-based network showing robust local feature matching, outperforming the state-of-the-art methods on several benchmarks.	Paper, Tweet	
10) Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement - addresses the time series forecasting problem with generative modeling; involves a bidirectional VAE backbone equipped with diffusion, denoising for prediction accuracy, and disentanglement for model interpretability.	Paper, Tweet	


Top ML Papers of the Week (Jan 1-8)


Paper	Links	
1) Muse: Text-To-Image Generation via Masked Generative Transformers - introduces Muse, a new text-to-image generation model based on masked generative transformers; significantly more efficient than other diffusion models like Imagen and DALLE-2.	Paper, Project, Code, Tweet	
2) VALL-E Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers - introduces VALL-E, a text-to-audio model that performs state-of-the-art zero-shot performance; the text-to-speech synthesis task is treated as a conditional language modeling task.	Project, Tweet	
3) Rethinking with Retrieval: Faithful Large Language Model Inference - shows the potential of enhancing LLMs by retrieving relevant external knowledge based on decomposed reasoning steps obtained through chain-of-thought prompting.	Paper, Tweet	
4) SparseGPT: Massive Language Models Can Be Accurately Pruned In One-Shot - presents a technique for compressing large language models while not sacrificing performance; "pruned to at least 50% sparsity in one-shot, without any retraining."	Paper, Tweet	
5) ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders - a performant model based on a fully convolutional masked autoencoder framework and other architectural improvements. CNNs are sticking back!	Paper, Code, Tweet	
6) Large Language Models as Corporate Lobbyists - with more capabilities, we are starting to see a wider range of applications with LLMs. This paper utilized large language models for conducting corporate lobbying activities.	Paper , Code, Tweet	
7) Superposition, Memorization, and Double Descent - aims to better understand how deep learning models overfit or memorize examples; interesting phenomena observed; important work toward a mechanistic theory of memorization.	Paper, Tweet	
8) StitchNet: Composing Neural Networks from Pre-Trained Fragments - new idea to create new coherent neural networks by reusing pretrained fragments of existing NNs. Not straightforward but there is potential in terms of efficiently reusing learned knowledge in pre-trained networks for complex tasks.	Paper, Tweet	
9) Iterated Decomposition: Improving Science Q&A by Supervising Reasoning Processes - proposes integrated decomposition, an approach to improve Science Q&A through a human-in-the-loop workflow for refining compositional LM programs.	Paper, Code Tweet	
10) A Succinct Summary of Reinforcement Learning - a nice overview of some important ideas in RL.	Paper, Tweet	







































































































