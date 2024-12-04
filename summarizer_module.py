from openai import AzureOpenAI

def generate_summary_with_gpt(summaries):
    """
    Generate a cohesive summary from multiple summaries using GPT-4 on Azure OpenAI.

    Args:
        summaries (list of str): List of document summaries.

    Returns:
        str: The final summarized text.
    """
    # Define Azure OpenAI client
    oiclient = AzureOpenAI(
        azure_endpoint="https://openaiforchandrahas2.openai.azure.com/",  # Replace with your Azure OpenAI endpoint
        api_key="DPLqEJbmD3X0980uiCQfoYFvsQ9TpbPTzax8t5Q460sbJtOdTIToJQQJ99AKACYeBjFXJ3w3AAABACOGCF5E",  # Replace with your actual Azure OpenAI API key
        api_version="2024-08-01-preview"  # Replace with the correct API version
    )

    model = "gpt-4o"  # Replace with your Azure OpenAI model deployment name

    # Construct the prompt using summaries
    combined_input = "Summarize the following information shortly within 300 tokens:\n"
    for i, summary in enumerate(summaries, 1):
        combined_input += f"{i}. {summary}\n"

    # Call the Azure OpenAI Chat Completion API
    response = oiclient.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": combined_input,
            },
        ],
        max_tokens=500,
        temperature=0.2,
        top_p=1.0
    )

    # Extract the summarized text from the response object
    summarized_text = response.choices[0].message.content
    return summarized_text.strip()

def summarize_documents(documents):
    """
    Wrapper for generate_summary_with_gpt to maintain compatibility with app.py.

    Args:
        documents (list of str): List of document texts to summarize.

    Returns:
        str: Final summarized text.
    """
    return generate_summary_with_gpt(documents)