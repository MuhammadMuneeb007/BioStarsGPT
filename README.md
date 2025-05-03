# BioStarGPT

BioStarGPT is a project that extracts, processes, and generates questions and answers from the Biostars online forum, a popular platform for bioinformatics and computational biology Q&A. The primary purpose is to create high-quality training data for fine-tuning large language models (LLMs) on bioinformatics and computational biology content.

## Project Overview

The project consists of three main steps:

1. **Data Collection**: Scraping questions from the Biostars website
2. **Data Processing**: Downloading and processing individual questions with their answers
3. **Question Generation**: Filtering and generating structured question-answer pairs for LLM fine-tuning
   biostars_eda_20250403.png

## Why Fine-Tune LLMs on Biostars Data?

Biostars contains domain-specific knowledge and expert-validated solutions for bioinformatics problems that are valuable for training specialized AI models. By fine-tuning LLMs on this data, we aim to:

- Enhance model capabilities in bioinformatics and computational biology domains
- Improve understanding of field-specific terminology, tools, and methods
- Enable more accurate responses to bioinformatics queries
- Create AI systems that can assist researchers with specialized bioinformatics tasks

## Step 1: Download Questions List from BioStars

### Script: `Step1-GetQuestionsListFromBioStars.py`

This script scrapes the Biostars website to collect a list of questions with metadata.

#### Usage:

```bash
python Step1-GetQuestionsListFromBioStars.py
```

#### Output:

```bash
biostars_all_questions.csv
```

The script generates a CSV file containing information about questions from BioStars, including:

| QuestionCount | Votes | Replies | Views | Title                                                                                                     | URL                                 | Tags                                     | Updated_Year |
| ------------- | ----- | ------- | ----- | --------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------- | ------------ |
| 1             | 0     | 1       | 222   | Question about `vg construct`                                                                             | https://www.biostars.org/p/9610704/ | vg                                       |              |
| 2             | 0     | 0       | 45    | How can I use mcmcglmm to calculate whether two traits are correlated?                                    | https://www.biostars.org/p/9610892/ | glmm\|statistics\|evolution              |              |
| 3             | 0     | 1       | 140   | DiffBind normalization error: invalid argument type (list) - cannot make it work everything seems correct | https://www.biostars.org/p/9610880/ | normalization\|deseq2\|chipseq\|diffbind |              |

### Script: `Step1.1-PerformEDAOnQuestionsFromBioStars.py`

Performs analysis on the questions to identify the most common tags and words in the question titles.

#### Usage:

```bash
python Step1.1-PerformEDAOnQuestionsFromBioStars.py
```

#### Output

The script generates exploratory data analysis on the questions, including common tags and word frequencies:

![Analysis of Questions from BioStars](Figure1-AnalysisOfQuestions.png)

This analysis helps understand the content distribution and common topics in the BioStars dataset.

## Step 2: Filter Questions Based on Replies

### Script: `Step2-FilterQuestionsFromBioStars.py`

This script filters questions from the dataset based on reply count criteria.

#### Filtering Criteria:

- Minimum votes: 0
- Minimum replies: 1

#### Usage:

```bash
python Step2-FilterQuestionsFromBioStars.py
```

#### Output

```bash
biostars_filtered_questions.csv
```

The script generates a CSV file containing filtered questions that meet the criteria:

| QuestionCount | Votes | Replies | Views | Title                                                                                                     | URL                                 | Tags                                     | Updated_Year | replies |
| ------------- | ----- | ------- | ----- | --------------------------------------------------------------------------------------------------------- | ----------------------------------- | ---------------------------------------- | ------------ | ------- |
| 1             | 0.0   | 1       | 222.0 | Question about `vg construct`                                                                             | https://www.biostars.org/p/9610704/ | vg                                       |              | 1       |
| 3             | 0.0   | 1       | 140.0 | DiffBind normalization error: invalid argument type (list) - cannot make it work everything seems correct | https://www.biostars.org/p/9610880/ | normalization\|deseq2\|chipseq\|diffbind |              | 1       |
| 4             | 1.0   | 7       | 1.4   | FastQC on nanopore data: high proportion of polyA and polyG. Why ?                                        | https://www.biostars.org/p/9589811/ | fastQC\|polyG\|ONT\|nanopore\|polyA      |              | 7       |

## Step 3: Download Each Question Data

### Script: `Step3-DownloadQuestionsData.py`

This script downloads and processes individual questions, their answers, and images. It creates a directory for Questions, and for each question, it creates a subdirectory named after the question ID (e.g., 1, 2, 3, 4, 5), and downloads the HTML content.

- Extracts full question content, answers, and comments
- Downloads and processes images from posts
- Performs OCR on images to extract text content
- Creates both raw and processed markdown files

#### Usage:

```bash
python Step3-DownloadQuestionsData.py
```

#### Output:

```
Scraping question: https://www.biostars.org/p/9484424/
Created directory: Questions/1
Created directory: Questions/1/images
Downloaded image: Questions/1/images/example_image.png
Extracted text from image: [OCR Text: sequence alignment visualization]
Saved question details to Questions/1/Text.md
Saved processed markdown to Questions/1/ProcessedText.md
```

## Step 4: Generate Question/Answer Pairs

### Script: `Step4-GenerateQuestionsLinear.py`

This script filters questions based on criteria and prepares prompts for generating structured question-answer pairs.

#### Process:

- Filters questions based on votes, replies, or tags
- Transforms processed HTML to question-answer pairs using Google Gemini
- Utilizes careful prompt engineering to improve model responses
- Processes 90,000+ questions using 12 threads (taking approximately 1 month)
- Requires Google Gemini API access (used models: "gemini-2.5-flash-preview-04-17" and "gemini-2.0-flash-thinking-exp-01-21") Visit the [Google AI Studio](https://aistudio.google.com/)
- Formats questions into JSON format

For more information on the Gemini API, visit the [official documentation](https://ai.google.dev/docs/gemini_api_overview).

#### Prompt Example:

````python
prompt = f"""
You are an expert in extracting questions and answers from Biostars. Your task is to extract
questions and answers from the Biostars online forum, where users post queries, problems, and images,
and others respond in the form of comments. Some comments may contain accurate answers, and those answers have high upvotes.

  ### **1. Extraction Rules**
  - **Ignore** usernames.
  - **Include** references to external tools or links.
  - **Ignore** the number of likes or dislikes.
  - **Choose** the answer based on the comment with the highest number of upvotes and if upvotes are missing then choose the most appropriate answer based on the discussion.

  - **Provide** the source link for each answer.
  - **If no valid answer is found,** use "-" in the answer section.
  - **Limit to two questions only.**
  - **Use direct instructions** (e.g., "How to do something?" instead of "How does one do something?").
  - **Ensure answers follow an instructional style** (e.g., "To do this, follow these steps...").
  - **Use a conversational tone** (e.g., "You can do this by...") to make the content more engaging. and do not use "I" or "we" in the answer or the user is asking or the user wants.

  ### **2. Output Format**
  - Format the extracted data as a **valid JSON array**:

  ```json
  [
      {{
    "Question1": "General question title?",
    "Explanation": "General explanation based on the discussion overall discussion", 100+ words based on discussion.
    "Source": "{row['URL']}",
    "Answer":YES/NO, **If there is no answer found in explanation put no otherwise yes. **
    "Tags": ["tag1", "tag2"]
      }},
      {{
    "Question2": "Detailed question based on the issue the user is facing?", **it should be a detailed questions based on the user query.**
    "Explanation2": "Detailed explanation of the problem and its solution.", **500+ words based on discussion. Include code and explanation as well**.
    "Source": "{row['URL']}",
    "Answer":YES/NO, **If there is no answer found in explanation put no otherwise yes. **
    "Tags": ["tag1", "tag2"]
      }}
  ]


### **3. Formatting Guidelines**

- Use **Markdown triple backticks** (```) for **code snippets**.
- If applicable, use **tables** for clarity.
- Maintain **proper indentation and spacing** for readability.

### **4. Strict Adherence to Provided Information**

- **Do not fabricate** any details not explicitly found in the provided text.
- Stay strictly within the **scope of the given content**.

### **5. Citations & References**

- At the end of each complete answer, **cite the source** as follows:

`SOURCE LINK: {row['URL']}`

---

### **Text from Discussion:**

{content}

---

"""
````

#### Usage:

```bash
python Step4-GenerateQuestionsLinear.py
```

#### Dataset Overview:

This generates dataset comprises questions and answers extracted from Biostars forum posts. The following steps were taken to process and enhance the content:

- Each question was downloaded in HTML format
- Images within posts were processed using OCR, and the extracted text was appended to the question content
- A Markdown version of each question was created and saved
- The Markdown version was submitted to Google Gemini to generate two questions per post:
  - **Question 1**: A general summary of the user's issue
  - **Question 2**: A detailed technical restatement of the issue

Each generated question is paired with an explanation and metadata based on the original Biostars post.

The dataset is available on HuggingFace: [BioStarsDataset](https://huggingface.co/datasets/muhammadmuneeb007/BioStarsDataset/edit/main/README.md)

##### Field Descriptions

- **Question1**: A general or high-level version of the question
- **Explanation1**: A summarized response generated by Google Gemini
- **Source**: URL linking to the original Biostars post
- **Answer**: `"YES"` if the AI found an answer, `"NO"` if no answer was found
- **Tags**: Topics associated with the question
- **QuestionCount**: Number of generated questions
- **URL**: Redundant copy of the source link
- **Votes**: Vote count on the original post
- **Views**: Number of views the post received
- **Replies**: Number of replies to the post
- **Title**: Title of the Biostars post
- **Question2**: A more technical or detailed rephrasing of the original issue
- **Explanation2**: A comprehensive answer to the detailed question
- **Answer2**: The model's full, detailed answer

> ⚠️ Note: The `Answer` fields reflect whether the model successfully generated a meaningful response (`YES`) or not (`NO`).
> This dataset comprises questions and answers extracted from Biostars forum posts. The following steps were taken to process and enhance the content:

- Each question was downloaded in HTML format
- Images within posts were processed using OCR, and the extracted text was appended to the question content
- A Markdown version of each question was created and saved
- The Markdown version was submitted to Google Gemini to generate two questions per post:
  - **Question 1**: A general summary of the user's issue
  - **Question 2**: A detailed technical restatement of the issue

Each generated question is paired with an explanation and metadata based on the original Biostars post.

##### Field Descriptions

- **Question1**: A general or high-level version of the question
- **Explanation1**: A summarized response generated by Google Gemini
- **Source**: URL linking to the original Biostars post
- **Answer**: `"YES"` if the AI found an answer, `"NO"` if no answer was found
- **Tags**: Topics associated with the question
- **QuestionCount**: Number of generated questions
- **URL**: Redundant copy of the source link
- **Votes**: Vote count on the original post
- **Views**: Number of views the post received
- **Replies**: Number of replies to the post
- **Title**: Title of the Biostars post
- **Question2**: A more technical or detailed rephrasing of the original issue
- **Explanation2**: A comprehensive answer to the detailed question
- **Answer2**: The model's full, detailed answer

> ⚠️ Note: The `Answer` fields reflect whether the model successfully generated a meaningful response (`YES`) or not (`NO`).

#### Sample Output:

````json
{
  "Question1": "Why does `vg construct --alt-paths-plain` cause errors?",
  "Explanation1": "The user is experiencing issues when using the `vg construct` command with the `--alt-paths-plain` option for constructing a variation graph. They are encountering warnings indicating 'missingempty' paths and a mismatch between variants in the VCF file and the constructed graph. The core problem is a mismatch between how variant paths are named when using `--alt-paths-plain` and what the GBWT indexing process expects. This option leads to naming variant paths using variant IDs from the VCF, which is incompatible with GBWT indexing that relies on SHA1 hashes of variants for path naming. Consequently, GBWT indexing fails to locate the correct paths, resulting in errors and warnings during the indexing step.",
  "Source": "https://www.biostars.org/p/9610704/",
  "Answer": "YES",
  "Tags": [
    "vg"
  ],
  "QuestionCount": "1",
  "URL": "https://www.biostars.org/p/9610704/",
  "Votes": 0.0,
  "Views": 222.0,
  "Replies": 1,
  "Title": "Question about vg construct"
},
{
  "Question2": "How to resolve 'missingempty' path errors when using `vg construct --alt-paths-plain` and use VCF ID column instead of SHA1 hash?",
  "Explanation2": "The 'missingempty' path errors encountered when using `vg construct --alt-paths-plain` arise because the GBWT indexing process, used in commands like `vg index -G test.gbwt`, expects variant paths to be named based on the SHA1 hash of the variant, not the VCF ID. The `--alt-paths-plain` option instructs `vg construct` to name variant paths using the IDs directly from the VCF file. This naming convention is incompatible with GBWT, which is designed to work with hash-based path names. \n\n**Solution:**\n\nTo resolve this issue and avoid the errors, you should **remove the `--alt-paths-plain` option** from your `vg construct` command. By default, or when using `--alt-paths` (which is often implicit), `vg construct` will name variant paths using the SHA1 hash of the variant. This hash-based naming is compatible with the GBWT indexing process. \n\n```bash\n# Corrected vg construct command:\nvg construct --reference ref.fa --vcf test.vcf.gz --alt-paths --handle-sv --flat-alts --progress -t 100 > test.vg\n```\n\n**Explanation of why this works:**\n\n- **`--alt-paths` (or default behavior):** `vg construct` names variant paths using SHA1 hashes. GBWT indexing is designed to find and use these hash-named paths to reconstruct haplotypes.\n- **`--alt-paths-plain`:** `vg construct` names variant paths using VCF IDs. GBWT indexing cannot find paths named with VCF IDs, leading to the 'missingempty' errors.\n\n**Using VCF ID for representation:**\n\nDirectly using the VCF ID column for variant path naming within the standard `vg construct` and GBWT indexing workflow is **not recommended** and leads to errors as demonstrated. The GBWT indexing is fundamentally built around hash-based path naming for efficiency and consistency. If you specifically need to reference variants by their VCF IDs, you might need to explore alternative indexing strategies or graph traversal methods within the `vg` ecosystem that are compatible with ID-based referencing. However, for standard graph construction and indexing workflows with `vg` and GBWT, using the default or `--alt-paths` (hash-based naming) is the correct approach. If you require VCF IDs for downstream analysis, you would typically map back from the graph paths (which are internally hash-based) to the original VCF records based on variant positions and alleles, rather than altering the core path naming within `vg` and GBWT.\n\nIn summary, remove `--alt-paths-plain` and use `--alt-paths` (or default) to ensure compatibility with GBWT indexing and resolve the 'missingempty' path errors. Using VCF IDs directly as path names in GBWT indexing is not supported and is the root cause of the problem when `--alt-paths-plain` is used.",
  "Source": "https://www.biostars.org/p/9610704/",
  "Answer": "YES",
  "Tags": [
    "vg"
  ],
  "QuestionCount": "1",
  "URL": "https://www.biostars.org/p/9610704/",
  "Votes": 0.0,
  "Views": 222.0,
  "Replies": 1,
  "Title": "Question about vg construct"
}
````

#### Directory Structure

After running the scripts, your project directory will look like:

```
BioStarGPT/
└── Questions/
    ├── 1/
    │   ├── Text.md
    │   ├── ProcessedText.md
    │   ├── raw.html
    │   ├── prompt.txt
    │   ├── Questions.json (Output of the Model)
    │   └── [images]
    ├── 2/
    │   └── ...
    └── ...
```

### Script: `Step4.1-VerifyQuestionsContent.py`

This script filters questions based on criteria and prepares prompts for generating structured question-answer pairs.

#### Process:

### Usage

To check content alignment between references and predictions:

```bash
python content_alignment_check.py
```

## Semantic Analysis

### Understanding Semantic Similarity vs. Content Alignment

#### What Semantic Similarity Measures

The semantic similarity score calculated by our models (ranging from 0 to 1) indicates how similar the **meaning** of two pieces of text are. A high score suggests that both texts are discussing the same concepts using similar language patterns and semantics.

However, it's important to understand what this score captures and what it doesn't:

##### What it DOES capture:

- Overall topic similarity
- Use of related terminology
- Similar context and domain references
- General alignment in discussion points

##### What it may NOT fully capture:

- Factual correctness of the claims in the prediction
- Whether the prediction actually answers the question asked
- If the prediction references the correct source article
- Technical accuracy of specific details

#### Content Alignment Check

The `content_alignment_check.py` script provides a deeper analysis by:

1. Calculating semantic similarity using the sentence transformer model
2. Checking if responses cite the same source URLs
3. Identifying cases where responses may seem semantically similar but are referring to different content

The script categorizes responses as "aligned" or "misaligned" based on a similarity threshold (default: 0.7) and provides tools to review potentially misaligned responses.

### Available Scripts

- `semanticanalysis.py`: Calculates semantic similarity between references and predictions across all epoch files
- `compare_models.py`: Benchmarks different semantic similarity models to find the optimal one for the BioStarsGPT data
- `content_alignment_check.py`: Analyzes whether references and predictions are referring to the same content/topic

### Usage

To check content alignment between references and predictions:

```bash
python content_alignment_check.py
```

To analyze semantic similarity across all epochs:

```bash
python semanticanalysis.py
```

To compare different semantic similarity models:

```bash
python compare_models.py
```

### Interpreting Results

When reviewing semantic similarity scores:

- **High score (>0.8)**: Strong alignment in meaning and content
- **Medium score (0.6-0.8)**: Moderate alignment, may differ in details or emphasis
- **Low score (<0.6)**: Significant divergence in content or meaning

For content alignment analysis, review the `misaligned_examples.json` file to understand cases where predictions may appear semantically similar but refer to different content than the reference.

## Tag Filtering Tool

### Script: `Step6.2-GetTags.py`

This tool processes BioStars data (both Train.json and Test.json) and filters questions by specific tags, saving the results as Train.jsonl and Test.jsonl.

#### Usage

##### Using predefined tags in the script:

You can directly modify the `predefined_tags` list in the `main` function:

```python
# Inside Step6.2-GetTags.py
def main():
    # Modify this list as needed
    predefined_tags = ["plink", "gwas", "genetics", "r"]
    # ...rest of the function...
```

Then simply run without arguments:

```bash
python Step6.2-GetTags.py
```

##### Overriding predefined tags via command line:

```bash
python Step6.2-GetTags.py --tags RNA-Seq alignment python
```

This will create two files:

- `Train.jsonl`: Contains filtered questions from Train.json that have any of the specified tags
- `Test.jsonl`: Contains filtered questions from Test.json that have any of the specified tags

#### Notes

- Tag filtering is case-insensitive
- All tags are converted to lowercase in the output
- The output is in JSONL format (one JSON object per line)
- Each entry contains:
  - `prompt`: The question
  - `completion`: The explanation with SOURCE LINKS removed
  - `tags`: The original tags in lowercase

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
