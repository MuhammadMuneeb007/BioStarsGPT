# BioStarGPT

BioStarGPT is a project that extracts, processes, and generates questions and answers from the Biostars online forum, a popular platform for bioinformatics and computational biology Q&A.

## Project Overview

The project consists of three main steps:

1. **Data Collection**: Scraping questions from the Biostars website
2. **Data Processing**: Downloading and processing individual questions with their answers
3. **Question Generation**: Filtering and generating structured question-answer pairs

## Prerequisites

- Python 3.8+
- Required packages (can be installed via `pip install -r requirements.txt`):
  - requests
  - beautifulsoup4
  - pandas
  - html2markdown
  - easyocr
  - numpy
  - opencv-python
  - Pillow
  - markdownify
  - markdown2

## Step 1: Collecting Questions List

### Script: `Step1-GetQuestionsList.py`

This script scrapes the Biostars website to collect a list of questions with metadata.

#### Features:
- Multi-process scraping for faster data collection
- Collects votes, replies, views, titles, URLs, and tags
- Saves data to CSV format

#### Usage:
```bash
python Step1-GetQuestionsList.py
```

#### Sample Output:
```
Scraping Biostars questions using multiprocessing...
Distributing work across 5 processes:
Process 1 will handle pages 1 to 484
Process 2 will handle pages 485 to 968
Process 3 will handle pages 969 to 1452
Process 4 will handle pages 1453 to 1936
Process 5 will handle pages 1937 to 2418
Process 30260: Scraping page 1...
Process 30262: Scraping page 969...
Process 30261: Scraping page 485...
Process 30263: Scraping page 1453...
Process 30264: Scraping page 1937...
Found 48360 questions
Data saved to C:\Users\kl\Desktop\The University of Queensland\BioStarGPT\biostars_questions.csv
```

## Step 2: Processing Individual Questions

### Script: `Step1-GetQuestions.py`

This script downloads and processes individual questions, their answers, and images.

#### Features:
- Extracts full question content, answers, and comments
- Downloads and processes images from posts
- Performs OCR on images to extract text content
- Creates both raw and processed markdown files
- Fixes image references in markdown files

#### Usage:
```bash
python Step1-GetQuestions.py
```

To fix image references in existing markdown files:
```bash
python Step1-GetQuestions.py --fix-markdown
```

#### Sample Output:
```
Scraping question: https://www.biostars.org/p/9484424/
Created directory: Questions/1
Created directory: Questions/1/images
Downloaded image: Questions/1/images/example_image.png
Extracted text from image: [OCR Text: sequence alignment visualization]
Saved question details to Questions/1/Text.md
Saved processed markdown to Questions/1/ProcessedText.md
```

## Step 3: Generating Structured Questions

### Script: `Step2-GenerateQuestions.py`

This script filters questions based on criteria and prepares prompts for generating structured question-answer pairs.

#### Features:
- Filters questions based on votes, replies, or tags
- Creates prompts for extracting structured information
- Processes questions into a JSON format

#### Usage:
```bash
python Step2-GenerateQuestions.py
```

#### Sample Output:
```
Loaded 48360 questions from CSV
Found 235 questions matching the criteria.
Processing question 50...
Created prompt file: Questions/50/prompt.txt
```

## Generated Question-Answer Format

The system generates questions and answers in a structured JSON format:

```json
[
    {
        "Question1": "How do I perform differential gene expression analysis?",
        "Explanation": "A detailed explanation of differential gene expression analysis concepts...",
        "Source": "https://www.biostars.org/p/123456/",
        "Answer": "YES",
        "Tags": ["RNA-Seq", "differential-expression", "DESeq2"]
    },
    {
        "Question2": "How can I normalize RNA-seq data for batch effects using DESeq2?",
        "Explanation2": "A comprehensive explanation including code samples for normalizing RNA-seq data...",
        "Source": "https://www.biostars.org/p/123456/",
        "Answer": "YES",
        "Tags": ["RNA-Seq", "differential-expression", "DESeq2", "batch-effect"]
    }
]
```

## Directory Structure

After running the scripts, your project directory will look like:

```
BioStarGPT/
├── Step1-GetQuestionsList.py
├── Step1-GetQuestions.py
├── Step2-GenerateQuestions.py
├── biostars_questions.csv
├── prompt.txt
└── Questions/
    ├── 1/
    │   ├── Text.md
    │   ├── ProcessedText.md
    │   ├── raw.html
    │   ├── prompt.txt
    │   └── [images]
    ├── 2/
    │   └── ...
    └── ...
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
