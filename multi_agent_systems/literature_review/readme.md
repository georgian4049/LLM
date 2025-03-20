# Literature Review Generation with Multi-Agent Systems

This project is designed to generate literature reviews on a given topic using a multi-agent system. The system consists of several agents that work together to download, process, summarize, and generate a coherent literature review based on research papers related to the specified topic.

## Features

- **Topic Input**: The user can input a topic for which they want to generate a literature review.
- **Research Paper Downloading**: The system can download and process research papers from arXiv based on the topic.
- **Summarization**: The system summarizes the content of the research papers.
- **Literature Review Generation**: The system generates a coherent literature review based on the summaries of the research papers.
- **Feedback Mechanism**: The system allows for user feedback to improve the generated literature review.

## Agents

1. **Download_and_Process_Agent**: Downloads and processes research papers from arXiv.
2. **Summary_Agent**: Summarizes the content of the research papers.
3. **Literature_Review_Agent**: Generates a literature review based on the summaries.
4. **Feedback_Agent**: Processes user feedback and suggests improvements to the literature review.

## Usage

1. **Install Dependencies**: Install the required dependencies using the `requirements.txt` file.
2. **Run the Notebook**: Open the `final_portfolio.ipynb` notebook and run the cells to generate the literature review.
3. **Input Topic**: When prompted, input the topic for which you want to generate the literature review.
4. **Review and Feedback**: Review the generated literature review and provide feedback if necessary.

## Requirements

To run this project, you need to install the following dependencies:

```bash
pip install -r requirements.txt