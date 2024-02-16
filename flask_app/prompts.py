DEFAULT_COSTS_PROMPT = f"""
    You will be given a document containing details about a project, and the costs associated
    with the project. Your goal is to analyze this text, and give a detailed response that 
    outlines all the costs. Include specific numbers and ensure that no important information is lost.
    Do not hallucinate your response, and make sure you only use information from the document provided.
    Document:
"""

DEFAULT_REQUIREMENTS_PROMPT = f"""
    You will be given a document containing details about a project. Your goal is to analyze this text, 
    and give a detailed response that outlines the requirements of the project. Your response should
    include specific details about the requirements. If something is vague or ambiguous, do not make
    assumptions, instead explain why it is vague or ambiguous. Do not hallucinate your response, and make sure you
    only use information from the document provided.
    Document:
"""

DEFAULT_STAKEHOLDERS_PROMPT = f"""
    You will be given a document containing details about a project. Your goal is to analyze this text, 
    and give a detailed response that lists all the stakeholders in the project. Your response should
    include specific details such as their names, and the roles they encapsulate. Ensure your response
    includes the stakeholder's relation to the project. Do not hallucinate your response, and make sure you
    only use information from the document provided.
    Document:
"""

DEFAULT_PROMPT_MAPPING = {
    "Costs": DEFAULT_COSTS_PROMPT,
    "Requirements": DEFAULT_REQUIREMENTS_PROMPT,
    "Stakeholders": DEFAULT_STAKEHOLDERS_PROMPT
}