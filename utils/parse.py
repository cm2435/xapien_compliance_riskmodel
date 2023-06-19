import time

import openai


def invoke_chatgpt(prompt):
    openai.api_key = "" # Replace with your actual API key

    retry_count = 0
    max_retries = 5
    backoff_time = 2  # seconds

    while retry_count < max_retries:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ]
            )

            if len(response.choices) > 0:
                return response.choices[0].message.content.strip()

            return None
        except openai.error.RateLimitError:
            retry_count += 1
            time.sleep(backoff_time)
            backoff_time *= 2  # Exponential backoff

    return None


def generate_prompt(title, snippet):
    prompt = f"""You are a tool that is designed to analyse the risk of a piece of text data for a client. The data will be about a given company. The goal is to help a layperson be able to understand what a company may be involved with and when. It is vital we catch risky dealings of companies we analyse. 

If the company is mentioned in an article, it does not mean it is necessarily risky. For example, a fraud prosecuting law firm is not risky, a company being prosecuted for fraud is risky. 

Given a few hand labelled examples, I want you to classify if they indicate that the company is risky for an interested investor, and thus they should be shown it to do their due dilligance.
Generate NOTHING other than the number corresponding to 1 for risky and 0 for not 

EXAMPLES:
Title: Government ‘Close to Decriminalising Waste Crime and Fly-Tipping’, say MPs
Snippet: '19 Oct 2022 - HMRC spent six years investigating suspected £78m landfill tax evasion run by Niramax, a waste disposal company based in North East England, ...'
Label: 1 

Title: 'Zurich Insurance PLC v Niramax Group Limited [2021] ...'
Snippet: '5 May 2021 - ... Chambers Social Responsibility · Complaints Procedure ... In Zurich Insurance PLC v Niramax Group Limited, dealing with a contract ...'
Label: 0

Title: 'Underwriting on trial'
Snippet: '30 Jul 2021 - In the first case, Zurich Insurance plc v Niramax Group Ltd ... Niramax, the insured, was in the business of waste collection and recycling.'
Label:0

Title: 'Millionaire jailed for orchestrating death of man he thought ...'
Snippet: '5 Mar 2020 - Neil Elliott organised a fatal attack on innocent Michael Phillips ... killing of Michael Phillips in Hartlepool Niramax boss Neil Elliott.'
Label:1 

REAL EXAMPLE: 
Title: {title}
Snippet:{snippet}
Label:
"""
    return prompt