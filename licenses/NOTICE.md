# Attribution and redistribution notes

Project source code is provided under the MIT license in the repository root. Dataset material retains its own terms and attribution; this project does not relicense it.

- **HumanEval** — OpenAI, Mark Chen et al., “Evaluating Large Language Models Trained on Code” (2021). Source: https://github.com/openai/human-eval . The upstream MIT notice is included as `HumanEval-MIT.txt`.
- **BigCodeBench** — Terry Yue Zhuo et al., “BigCodeBench: Benchmarking Code Generation with Diverse Function Calls and Complex Instructions” (2024). Source: https://huggingface.co/datasets/bigcode/bigcodebench . Dataset card declares Apache-2.0; upstream license is included as `BigCodeBench-Apache-2.0.txt`.
- **LiveCodeBench** — Naman Jain et al., “LiveCodeBench: Holistic and Contamination Free Evaluation of Large Language Models for Code” (2024). Source: https://huggingface.co/datasets/livecodebench/code_generation_lite . The dataset card is included unchanged. Its metadata says `license: cc` without a specific Creative Commons variant; the loading script and GitHub code describe MIT. The included MIT file applies to repository code and must not be interpreted as clarification of all underlying competition-statement rights. Original platform authors retain their rights. This research subset preserves provenance and is not offered under a newly asserted license. Review upstream/platform terms before redistribution or uses beyond your experiment.

All LiveCodeBench public and private tests in this subset come from the Lite release, which itself samples/prunes the original benchmark's tests. No test outcomes or reference solutions were fabricated. Task dates and IDs are retained in `data/tasks.jsonl`. The project does not claim endorsement by benchmark authors or official leaderboard compatibility.

LiveCodeBench statement/test-derived prompt text and test bodies are not redistributed here. The records archive retains identifiers and SHA-256 placeholders for those fields. `run.py` restores the initial task inputs from the pinned source into the ignored `data` directory. Generated completions and evaluated outcomes are retained unchanged.

For replacement tasks in the saved collection, the archive's `protocol.json` records the pinned additional source files, URLs, task IDs and source/test hashes. Opaque encrypted provider payloads are represented by hashes; response text and programs are retained.
